//! Optical flow and warping for the temporal lock.
//!
//! ENHANCEMENTS.md Phase 1. The lock holds invented detail still across frames
//! by warping the previous *output* forward along motion estimated from the
//! *source* frames, so a strong detail model stops re-inventing texture every
//! frame.
//!
//! ## Why a built-in estimator rather than a RAFT ONNX
//!
//! The lock is not an optional extra the way interpolation is — without it this
//! is an image upscaler applied 24 times a second. Requiring a second model
//! download would mean most users never get the feature that closes the quality
//! gap. Flow quality is the obvious objection, and the answer is the confidence
//! gate: where the estimate is unreliable the pipeline falls back to fresh
//! detail, so bad flow degrades to today's behaviour rather than to ghosting.
//!
//! `estimate` is a single function and can be swapped for a learned model later
//! without touching the rest of the pipeline.
//!
//! ## Convention
//!
//! `estimate(prev, cur)` returns flow such that `prev(p + flow(p)) ≈ cur(p)`.
//! That direction is what lets `warp(prev_output, flow)` produce an estimate of
//! the current frame from the previous one.

use rayon::prelude::*;

use crate::media::Frame;

/// Motion vectors on a coarse grid, in full-resolution pixels.
///
/// Block resolution rather than per-pixel: the field is smooth at the scale
/// that matters for holding texture in place, and a dense field would cost far
/// more than the detail model it is protecting.
#[derive(Debug, Clone)]
pub struct Flow {
    pub cols: usize,
    pub rows: usize,
    /// Full-resolution pixels covered by one grid cell.
    pub cell: usize,
    pub dx: Vec<f32>,
    pub dy: Vec<f32>,
}

impl Flow {
    fn zeros(cols: usize, rows: usize, cell: usize) -> Self {
        Self {
            cols,
            rows,
            cell,
            dx: vec![0.0; cols * rows],
            dy: vec![0.0; cols * rows],
        }
    }

    /// Bilinearly samples the field at a full-resolution pixel.
    pub fn at(&self, x: f32, y: f32) -> (f32, f32) {
        if self.cols == 0 || self.rows == 0 {
            return (0.0, 0.0);
        }
        // Grid centres sit at (i + 0.5) * cell in pixel space.
        let gx = (x / self.cell as f32 - 0.5).clamp(0.0, (self.cols - 1) as f32);
        let gy = (y / self.cell as f32 - 0.5).clamp(0.0, (self.rows - 1) as f32);
        let x0 = gx.floor() as usize;
        let y0 = gy.floor() as usize;
        let x1 = (x0 + 1).min(self.cols - 1);
        let y1 = (y0 + 1).min(self.rows - 1);
        let tx = gx - x0 as f32;
        let ty = gy - y0 as f32;

        let lerp = |v: &[f32]| {
            let top = v[y0 * self.cols + x0] * (1.0 - tx) + v[y0 * self.cols + x1] * tx;
            let bot = v[y1 * self.cols + x0] * (1.0 - tx) + v[y1 * self.cols + x1] * tx;
            top * (1.0 - ty) + bot * ty
        };
        (lerp(&self.dx), lerp(&self.dy))
    }

    /// Mean vector magnitude, used to spot implausible global motion.
    pub fn mean_magnitude(&self) -> f32 {
        if self.dx.is_empty() {
            return 0.0;
        }
        let sum: f32 = self
            .dx
            .iter()
            .zip(&self.dy)
            .map(|(x, y)| (x * x + y * y).sqrt())
            .sum();
        sum / self.dx.len() as f32
    }
}

/// Single-channel image used for matching.
struct Plane {
    w: usize,
    h: usize,
    v: Vec<f32>,
}

impl Plane {
    fn from_frame(frame: &Frame) -> Self {
        let (w, h) = (frame.width as usize, frame.height as usize);
        let mut v = vec![0f32; w * h];
        v.par_iter_mut().enumerate().for_each(|(i, out)| {
            let p = i * 3;
            // Rec.601 luma: matching only needs brightness structure, and
            // dropping chroma cuts the work to a third.
            *out = 0.299 * frame.rgb[p] as f32
                + 0.587 * frame.rgb[p + 1] as f32
                + 0.114 * frame.rgb[p + 2] as f32;
        });
        Self { w, h, v }
    }

    /// Halves the resolution with a 2x2 box filter.
    fn halve(&self) -> Plane {
        let w = (self.w / 2).max(1);
        let h = (self.h / 2).max(1);
        let mut v = vec![0f32; w * h];
        v.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
            for (x, out) in row.iter_mut().enumerate() {
                let sx = (x * 2).min(self.w - 1);
                let sy = (y * 2).min(self.h - 1);
                let sx1 = (sx + 1).min(self.w - 1);
                let sy1 = (sy + 1).min(self.h - 1);
                *out = 0.25
                    * (self.v[sy * self.w + sx]
                        + self.v[sy * self.w + sx1]
                        + self.v[sy1 * self.w + sx]
                        + self.v[sy1 * self.w + sx1]);
            }
        });
        Plane { w, h, v }
    }

    #[inline]
    fn get(&self, x: isize, y: isize) -> f32 {
        let x = x.clamp(0, self.w as isize - 1) as usize;
        let y = y.clamp(0, self.h as isize - 1) as usize;
        self.v[y * self.w + x]
    }
}

/// Grid cell size in pixels at every pyramid level.
const BLOCK: usize = 8;
/// Search radius around the propagated estimate, in pixels, per level.
const RADIUS: isize = 4;
/// Stop coarsening once the image is about this small.
const MIN_SIDE: usize = 48;

/// Estimates motion from `prev` to `cur`.
///
/// Coarse-to-fine block matching: each level refines the level above, which is
/// what lets an 8px block track motion far larger than 8px without an
/// enormous search window.
pub fn estimate(prev: &Frame, cur: &Frame) -> Result<Flow, String> {
    if prev.width != cur.width || prev.height != cur.height {
        return Err("flow inputs differ in size".into());
    }
    if prev.width == 0 || prev.height == 0 {
        return Err("flow inputs are empty".into());
    }

    // Pyramid, finest first.
    let mut levels_prev = vec![Plane::from_frame(prev)];
    let mut levels_cur = vec![Plane::from_frame(cur)];
    while levels_prev.last().unwrap().w > MIN_SIDE && levels_prev.last().unwrap().h > MIN_SIDE {
        levels_prev.push(levels_prev.last().unwrap().halve());
        levels_cur.push(levels_cur.last().unwrap().halve());
        if levels_prev.len() >= 5 {
            break;
        }
    }

    let coarsest = levels_prev.len() - 1;
    let mut flow = {
        let p = &levels_prev[coarsest];
        Flow::zeros(p.w.div_ceil(BLOCK), p.h.div_ceil(BLOCK), BLOCK)
    };

    for level in (0..levels_prev.len()).rev() {
        let p = &levels_prev[level];
        let c = &levels_cur[level];
        let cols = p.w.div_ceil(BLOCK);
        let rows = p.h.div_ceil(BLOCK);

        if flow.cols != cols || flow.rows != rows {
            flow = upsample(&flow, cols, rows);
        }

        let (dx, dy): (Vec<f32>, Vec<f32>) = (0..cols * rows)
            .into_par_iter()
            .map(|index| {
                let bx = index % cols;
                let by = index / cols;
                let ox = (bx * BLOCK) as isize;
                let oy = (by * BLOCK) as isize;
                let seed_x = flow.dx[index].round() as isize;
                let seed_y = flow.dy[index].round() as isize;

                let mut best = f32::MAX;
                let mut best_v = (seed_x, seed_y);
                for cy in -RADIUS..=RADIUS {
                    for cx in -RADIUS..=RADIUS {
                        let vx = seed_x + cx;
                        let vy = seed_y + cy;
                        let cost = sad(c, p, ox, oy, vx, vy);
                        // Tie-break toward the smaller vector: on flat regions
                        // every offset matches equally well, and preferring
                        // zero keeps the field from turning into noise.
                        let biased = cost + 0.01 * ((vx * vx + vy * vy) as f32).sqrt();
                        if biased < best {
                            best = biased;
                            best_v = (vx, vy);
                        }
                    }
                }
                (best_v.0 as f32, best_v.1 as f32)
            })
            .unzip();

        flow.dx = dx;
        flow.dy = dy;
        median_filter(&mut flow);

        // Moving to the next finer level doubles the scale.
        if level > 0 {
            for v in flow.dx.iter_mut().chain(flow.dy.iter_mut()) {
                *v *= 2.0;
            }
        }
    }

    Ok(flow)
}

/// Sum of absolute differences between a block of `cur` and the same block of
/// `prev` displaced by (vx, vy).
#[inline]
fn sad(cur: &Plane, prev: &Plane, ox: isize, oy: isize, vx: isize, vy: isize) -> f32 {
    let mut total = 0.0;
    // Every other pixel: matching is dominated by this loop and the halved
    // sample count costs almost no accuracy at 8px blocks.
    let mut y = 0isize;
    while y < BLOCK as isize {
        let mut x = 0isize;
        while x < BLOCK as isize {
            let a = cur.get(ox + x, oy + y);
            let b = prev.get(ox + x + vx, oy + y + vy);
            total += (a - b).abs();
            x += 2;
        }
        y += 2;
    }
    total
}

/// Replaces each vector with the component-wise median of its 3x3 neighbourhood.
///
/// Block matching produces isolated wild vectors in flat or repetitive areas;
/// a median is the standard cheap way to remove them without blurring genuine
/// motion boundaries the way an average would.
fn median_filter(flow: &mut Flow) {
    if flow.cols < 3 || flow.rows < 3 {
        return;
    }
    let (cols, rows) = (flow.cols, flow.rows);
    let filter = |src: &[f32]| -> Vec<f32> {
        (0..cols * rows)
            .into_par_iter()
            .map(|index| {
                let x = (index % cols) as isize;
                let y = (index / cols) as isize;
                let mut window = [0f32; 9];
                let mut n = 0;
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        let sx = (x + dx).clamp(0, cols as isize - 1) as usize;
                        let sy = (y + dy).clamp(0, rows as isize - 1) as usize;
                        window[n] = src[sy * cols + sx];
                        n += 1;
                    }
                }
                let w = &mut window[..n];
                w.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                w[n / 2]
            })
            .collect()
    };
    flow.dx = filter(&flow.dx);
    flow.dy = filter(&flow.dy);
}

/// Resizes the grid, preserving vector scale.
fn upsample(flow: &Flow, cols: usize, rows: usize) -> Flow {
    let mut out = Flow::zeros(cols, rows, flow.cell);
    for y in 0..rows {
        for x in 0..cols {
            let sx = if cols > 1 {
                (x * flow.cols.saturating_sub(1)) as f32 / (cols - 1).max(1) as f32
            } else {
                0.0
            };
            let sy = if rows > 1 {
                (y * flow.rows.saturating_sub(1)) as f32 / (rows - 1).max(1) as f32
            } else {
                0.0
            };
            let x0 = (sx.floor() as usize).min(flow.cols.saturating_sub(1));
            let y0 = (sy.floor() as usize).min(flow.rows.saturating_sub(1));
            out.dx[y * cols + x] = flow.dx[y0 * flow.cols + x0];
            out.dy[y * cols + x] = flow.dy[y0 * flow.cols + x0];
        }
    }
    out
}

/// Samples `src` along `flow`, producing an estimate of the frame `flow` was
/// measured *to*.
///
/// `scale` maps the flow (measured at source resolution) onto a frame that may
/// be an upscaled output — the lock warps full-resolution detail using motion
/// estimated from the smaller source.
pub fn warp(src: &Frame, flow: &Flow, scale: f32) -> Frame {
    let (w, h) = (src.width as usize, src.height as usize);
    let mut rgb = vec![0u8; w * h * 3];

    rgb.par_chunks_mut(w * 3).enumerate().for_each(|(y, row)| {
        for x in 0..w {
            // Flow was measured at source scale; the frame may be larger.
            let (fx, fy) = flow.at(x as f32 / scale, y as f32 / scale);
            let sx = x as f32 + fx * scale;
            let sy = y as f32 + fy * scale;
            let (r, g, b) = sample(src, sx, sy);
            row[x * 3] = r;
            row[x * 3 + 1] = g;
            row[x * 3 + 2] = b;
        }
    });

    Frame {
        width: src.width,
        height: src.height,
        rgb,
    }
}

#[inline]
fn sample(frame: &Frame, x: f32, y: f32) -> (u8, u8, u8) {
    let (w, h) = (frame.width as usize, frame.height as usize);
    let x = x.clamp(0.0, (w - 1) as f32);
    let y = y.clamp(0.0, (h - 1) as f32);
    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let x1 = (x0 + 1).min(w - 1);
    let y1 = (y0 + 1).min(h - 1);
    let tx = x - x0 as f32;
    let ty = y - y0 as f32;

    let mut out = [0u8; 3];
    for (c, o) in out.iter_mut().enumerate() {
        let p = |px: usize, py: usize| frame.rgb[(py * w + px) * 3 + c] as f32;
        let top = p(x0, y0) * (1.0 - tx) + p(x1, y0) * tx;
        let bot = p(x0, y1) * (1.0 - tx) + p(x1, y1) * tx;
        *o = (top * (1.0 - ty) + bot * ty).clamp(0.0, 255.0) as u8;
    }
    (out[0], out[1], out[2])
}

/// Mean absolute per-channel difference, 0..255. The flicker metric.
pub fn mean_abs_diff(a: &Frame, b: &Frame) -> f32 {
    if a.rgb.len() != b.rgb.len() || a.rgb.is_empty() {
        return f32::MAX;
    }
    let sum: f64 = a
        .rgb
        .par_iter()
        .zip(b.rgb.par_iter())
        .map(|(x, y)| (*x as f64 - *y as f64).abs())
        .sum();
    (sum / a.rgb.len() as f64) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hash2(x: i32, y: i32) -> f32 {
        let mut n = (x as u32)
            .wrapping_mul(374_761_393)
            .wrapping_add((y as u32).wrapping_mul(668_265_263));
        n = (n ^ (n >> 13)).wrapping_mul(1_274_126_177);
        ((n ^ (n >> 16)) & 0xFFFF) as f32 / 65535.0
    }

    /// Smooth value noise at one spatial frequency.
    fn octave(x: f32, y: f32, period: f32) -> f32 {
        let (gx, gy) = (x / period, y / period);
        let (x0, y0) = (gx.floor(), gy.floor());
        let (tx, ty) = (gx - x0, gy - y0);
        let (sx, sy) = (tx * tx * (3.0 - 2.0 * tx), ty * ty * (3.0 - 2.0 * ty));
        let (xi, yi) = (x0 as i32, y0 as i32);
        let a = hash2(xi, yi);
        let b = hash2(xi + 1, yi);
        let c = hash2(xi, yi + 1);
        let d = hash2(xi + 1, yi + 1);
        let top = a + (b - a) * sx;
        let bot = c + (d - c) * sx;
        top + (bot - top) * sy
    }

    /// Multi-octave value noise, shifted by a known amount.
    ///
    /// Deliberately *not* per-pixel white noise: that has no structure above
    /// one pixel, so every level of a pyramid downsamples to flat grey and the
    /// coarse-to-fine search has nothing to lock onto. Real footage has
    /// structure at every scale, which is what makes a pyramid work at all.
    fn textured(w: u32, h: u32, shift_x: i32, shift_y: i32) -> Frame {
        let mut rgb = vec![0u8; (w * h * 3) as usize];
        for y in 0..h as i32 {
            for x in 0..w as i32 {
                let (sx, sy) = ((x - shift_x) as f32, (y - shift_y) as f32);
                let v = 0.55 * octave(sx, sy, 37.0)
                    + 0.30 * octave(sx, sy, 13.0)
                    + 0.15 * octave(sx, sy, 5.0);
                let i = ((y as u32 * w + x as u32) * 3) as usize;
                let level = (v * 255.0).clamp(0.0, 255.0) as u8;
                rgb[i] = level;
                rgb[i + 1] = level;
                rgb[i + 2] = level;
            }
        }
        Frame {
            width: w,
            height: h,
            rgb,
        }
    }

    #[test]
    fn static_scene_produces_near_zero_flow() {
        let a = textured(128, 128, 0, 0);
        let flow = estimate(&a, &a).expect("estimate");
        assert!(
            flow.mean_magnitude() < 0.5,
            "static scene drifted: {}",
            flow.mean_magnitude()
        );
    }

    #[test]
    fn warping_reduces_residual_on_translation() {
        // `cur` is `prev` shifted right and down by 3px.
        let prev = textured(160, 160, 0, 0);
        let cur = textured(160, 160, 3, 2);

        let flow = estimate(&prev, &cur).expect("estimate");
        let warped = warp(&prev, &flow, 1.0);

        let before = mean_abs_diff(&prev, &cur);
        let after = mean_abs_diff(&warped, &cur);
        assert!(
            after < before * 0.6,
            "warp did not track the shift: {before} -> {after}"
        );
    }

    #[test]
    fn flow_sampling_is_bounded_outside_the_grid() {
        let flow = Flow::zeros(4, 4, 8);
        assert_eq!(flow.at(-500.0, -500.0), (0.0, 0.0));
        assert_eq!(flow.at(1e6, 1e6), (0.0, 0.0));
    }

    #[test]
    fn mismatched_inputs_are_rejected() {
        let a = textured(32, 32, 0, 0);
        let b = textured(64, 32, 0, 0);
        assert!(estimate(&a, &b).is_err());
    }

    /// Real footage, which is the only measurement that decides whether the
    /// built-in estimator is good enough to keep.
    ///
    /// Reports residual reduction and cost against the 1.65 s/frame model
    /// budget from the Phase 0 baseline.
    ///
    /// `cargo test --release flow_tracks_real_frames -- --ignored --nocapture`
    #[test]
    #[ignore = "set KRAKEN_TEST_CLIP to measure flow on real consecutive frames"]
    fn flow_tracks_real_frames() {
        let path = std::env::var_os("KRAKEN_TEST_CLIP")
            .map(std::path::PathBuf::from)
            .expect("KRAKEN_TEST_CLIP is required");
        let clip = crate::media::probe(&path).expect("probe clip");
        let step = 1.0 / clip.fps.max(1.0);

        let mut worst = 0.0f32;
        for index in 0..6u32 {
            let prev = crate::media::frame_at(&clip, index as f64 * step, None).expect("prev");
            let cur =
                crate::media::frame_at(&clip, (index + 1) as f64 * step, None).expect("cur");

            let started = std::time::Instant::now();
            let flow = estimate(&prev, &cur).expect("estimate");
            let elapsed = started.elapsed();
            let warped = warp(&prev, &flow, 1.0);

            let before = mean_abs_diff(&prev, &cur);
            let after = mean_abs_diff(&warped, &cur);
            let reduction = if before > 0.0 {
                (1.0 - after / before) * 100.0
            } else {
                0.0
            };
            worst = worst.max(after / before.max(1e-6));

            println!(
                "pair {index}->{}: residual {before:.2} -> {after:.2} ({reduction:+.0}%) · \
                 flow {:.0}ms · mean motion {:.2}px",
                index + 1,
                elapsed.as_secs_f32() * 1000.0,
                flow.mean_magnitude()
            );
        }

        // Warping must not make frames *less* similar. A stricter bar belongs
        // on the full lock, where the confidence gate can reject bad flow;
        // here the only claim is that the raw estimate is directionally right.
        assert!(
            worst < 1.0,
            "warping increased residual on at least one pair (ratio {worst:.2})"
        );
    }
}
