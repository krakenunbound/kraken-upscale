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
//! ## Known failure modes, for the confidence gate to catch
//!
//! Measured on real footage (`SpaceVoyage_4`, +35% mean residual reduction over
//! 10 pairs spread across the clip), the two weakest results were:
//!
//! * **Brightness change.** SAD is not invariant to intensity, so on a fade or
//!   lighting shift no offset matches well and the field is unreliable. The
//!   first frame pair of the clip reduced residual by only 2% despite 3.6px of
//!   apparent motion.
//! * **Large or discontinuous motion.** At 9.8px mean motion the reduction fell
//!   to 17%: block matching is weakest at motion boundaries, and the median
//!   filter that removes outliers also smooths across genuine edges.
//!
//! Neither is worth fixing in the estimator. Both are cases where the lock must
//! *decline to lock* and use fresh detail instead.
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

/// Flow plus a per-cell measure of how far it can be trusted.
pub struct Motion {
    pub flow: Flow,
    /// 0..1 per grid cell. 1 means the warp is reliable here.
    pub confidence: Vec<f32>,
    /// A hard edit. Nothing before it should be carried across.
    pub cut: bool,
}

/// Residual above which a match is considered worthless, in 0..255 levels.
///
/// Good matches on real footage land at 2–4; a brightness change leaves no
/// good match anywhere and pushes every cell well past this.
const RESIDUAL_TOLERANCE: f32 = 16.0;

/// Forward–backward disagreement above which a cell is considered occluded,
/// in source pixels.
const CONSISTENCY_TOLERANCE: f32 = 2.5;

impl Motion {
    /// Bilinearly samples confidence at a full-resolution source pixel.
    pub fn confidence_at(&self, x: f32, y: f32) -> f32 {
        let f = &self.flow;
        if f.cols == 0 || f.rows == 0 {
            return 0.0;
        }
        let gx = (x / f.cell as f32 - 0.5).clamp(0.0, (f.cols - 1) as f32);
        let gy = (y / f.cell as f32 - 0.5).clamp(0.0, (f.rows - 1) as f32);
        let x0 = gx.floor() as usize;
        let y0 = gy.floor() as usize;
        let x1 = (x0 + 1).min(f.cols - 1);
        let y1 = (y0 + 1).min(f.rows - 1);
        let tx = gx - x0 as f32;
        let ty = gy - y0 as f32;
        let c = &self.confidence;
        let top = c[y0 * f.cols + x0] * (1.0 - tx) + c[y0 * f.cols + x1] * tx;
        let bot = c[y1 * f.cols + x0] * (1.0 - tx) + c[y1 * f.cols + x1] * tx;
        top * (1.0 - ty) + bot * ty
    }

    pub fn mean_confidence(&self) -> f32 {
        if self.confidence.is_empty() {
            return 0.0;
        }
        self.confidence.iter().sum::<f32>() / self.confidence.len() as f32
    }
}

/// Estimates motion and decides where it can be trusted.
///
/// Two independent checks, because they fail on different things:
///
/// * **Match residual** — how well the warped previous frame actually predicts
///   the current one, per cell. This is what catches a brightness change: a
///   fade leaves no good match anywhere, and the residual stays high no matter
///   which offset was chosen.
/// * **Forward–backward consistency** — flow measured both ways should cancel.
///   Where it does not, the pixel is occluded or the match was arbitrary. This
///   is what catches motion boundaries, which residual alone can miss when a
///   wrong match happens to score well.
pub fn analyze(prev: &Frame, cur: &Frame) -> Result<Motion, String> {
    let flow = estimate(prev, cur)?;

    if crate::interpolate::is_scene_cut(prev, cur) {
        let confidence = vec![0.0; flow.dx.len()];
        return Ok(Motion {
            flow,
            confidence,
            cut: true,
        });
    }

    let backward = estimate(cur, prev)?;
    let warped = warp(prev, &flow, 1.0);
    let residual = cell_residuals(&warped, cur, &flow);

    let cols = flow.cols;
    let confidence = (0..flow.dx.len())
        .into_par_iter()
        .map(|index| {
            let fx = flow.dx[index];
            let fy = flow.dy[index];

            // Where does this cell land in the current frame, and does the
            // reverse estimate bring it back?
            let cx = ((index % cols) as f32 + 0.5) * flow.cell as f32;
            let cy = ((index / cols) as f32 + 0.5) * flow.cell as f32;
            let (bx, by) = backward.at(cx + fx, cy + fy);
            let mismatch = ((fx + bx).powi(2) + (fy + by).powi(2)).sqrt();

            let consistency = 1.0 - (mismatch / CONSISTENCY_TOLERANCE).min(1.0);
            let quality = 1.0 - (residual[index] / RESIDUAL_TOLERANCE).min(1.0);

            // Multiplied, not averaged: either failing alone is enough to make
            // the warp untrustworthy, and averaging would let a good residual
            // mask an occlusion.
            (consistency * quality).clamp(0.0, 1.0)
        })
        .collect();

    Ok(Motion {
        flow,
        confidence,
        cut: false,
    })
}

/// Mean absolute difference between two frames, per flow grid cell.
fn cell_residuals(a: &Frame, b: &Frame, flow: &Flow) -> Vec<f32> {
    let w = a.width as usize;
    let h = a.height as usize;
    let cell = flow.cell;

    (0..flow.cols * flow.rows)
        .into_par_iter()
        .map(|index| {
            let x0 = (index % flow.cols) * cell;
            let y0 = (index / flow.cols) * cell;
            let x1 = (x0 + cell).min(w);
            let y1 = (y0 + cell).min(h);
            if x0 >= x1 || y0 >= y1 {
                return RESIDUAL_TOLERANCE;
            }

            let mut sum = 0.0f32;
            let mut n = 0u32;
            for y in y0..y1 {
                for x in x0..x1 {
                    let p = (y * w + x) * 3;
                    sum += (a.rgb[p] as f32 - b.rgb[p] as f32).abs();
                    n += 1;
                }
            }
            sum / n.max(1) as f32
        })
        .collect()
}

/// Holds invented detail still: blends the warped previous output into the
/// current detail wherever the motion estimate can be trusted.
///
/// `strength` is the user-facing lock amount; confidence scales it down per
/// region. `scale` maps source-resolution flow onto the upscaled frames.
pub fn lock(
    previous_output: &Frame,
    current_detail: &Frame,
    motion: &Motion,
    strength: f32,
    scale: f32,
) -> Frame {
    let strength = strength.clamp(0.0, 1.0);
    if motion.cut || strength <= 0.0 {
        return current_detail.clone();
    }
    if previous_output.width != current_detail.width
        || previous_output.height != current_detail.height
    {
        // A resolution change mid-stream means there is nothing valid to carry
        // forward.
        return current_detail.clone();
    }

    let warped = warp(previous_output, &motion.flow, scale);
    let w = current_detail.width as usize;
    let mut rgb = vec![0u8; current_detail.rgb.len()];

    rgb.par_chunks_mut(w * 3).enumerate().for_each(|(y, row)| {
        let sy = y as f32 / scale;
        for x in 0..w {
            let weight = strength * motion.confidence_at(x as f32 / scale, sy);
            let p = (y * w + x) * 3;
            for c in 0..3 {
                let fresh = current_detail.rgb[p + c] as f32;
                let held = warped.rgb[p + c] as f32;
                row[x * 3 + c] = (fresh + (held - fresh) * weight).clamp(0.0, 255.0) as u8;
            }
        }
    });

    Frame {
        width: current_detail.width,
        height: current_detail.height,
        rgb,
    }
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

    /// Uniformly brightens a frame, leaving all structure in place.
    fn brighten(frame: &Frame, amount: i32) -> Frame {
        Frame {
            width: frame.width,
            height: frame.height,
            rgb: frame
                .rgb
                .iter()
                .map(|v| (*v as i32 + amount).clamp(0, 255) as u8)
                .collect(),
        }
    }

    #[test]
    fn confident_on_a_clean_translation() {
        let prev = textured(160, 160, 0, 0);
        let cur = textured(160, 160, 3, 2);
        let motion = analyze(&prev, &cur).expect("analyze");
        assert!(!motion.cut);
        assert!(
            motion.mean_confidence() > 0.5,
            "clean translation should be trusted, got {}",
            motion.mean_confidence()
        );
    }

    /// The first measured failure mode: SAD is not intensity-invariant, so a
    /// fade leaves no good match and the gate must refuse to lock.
    #[test]
    fn brightness_change_collapses_confidence() {
        let prev = textured(160, 160, 0, 0);
        let cur = brighten(&prev, 40);
        let motion = analyze(&prev, &cur).expect("analyze");
        assert!(
            motion.mean_confidence() < 0.3,
            "a fade must not be locked, got {}",
            motion.mean_confidence()
        );
    }

    #[test]
    fn scene_cut_disables_the_lock_entirely() {
        let prev = Frame {
            width: 64,
            height: 64,
            rgb: vec![10; 64 * 64 * 3],
        };
        let cur = Frame {
            width: 64,
            height: 64,
            rgb: vec![245; 64 * 64 * 3],
        };
        let motion = analyze(&prev, &cur).expect("analyze");
        assert!(motion.cut, "opposite frames must register as a cut");

        // And the blend must pass current detail through untouched.
        let locked = lock(&prev, &cur, &motion, 1.0, 1.0);
        assert_eq!(locked.rgb, cur.rgb);
    }

    #[test]
    fn lock_holds_detail_on_a_static_scene() {
        // Same source, but the "detail" differs — as it would when a model
        // re-invents texture independently each frame.
        let source = textured(128, 128, 0, 0);
        let previous_output = textured(128, 128, 0, 0);
        let mut current_detail = previous_output.clone();
        for (i, v) in current_detail.rgb.iter_mut().enumerate() {
            *v = (*v as i32 + if i % 7 == 0 { 26 } else { -14 }).clamp(0, 255) as u8;
        }

        let motion = analyze(&source, &source).expect("analyze");
        let locked = lock(&previous_output, &current_detail, &motion, 1.0, 1.0);

        let unlocked_flicker = mean_abs_diff(&current_detail, &previous_output);
        let locked_flicker = mean_abs_diff(&locked, &previous_output);
        assert!(
            locked_flicker < unlocked_flicker * 0.5,
            "lock failed to suppress invented detail: {unlocked_flicker} -> {locked_flicker}"
        );
    }

    #[test]
    fn zero_strength_is_a_passthrough() {
        let a = textured(64, 64, 0, 0);
        let b = textured(64, 64, 2, 0);
        let motion = analyze(&a, &b).expect("analyze");
        assert_eq!(lock(&a, &b, &motion, 0.0, 1.0).rgb, b.rgb);
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

        // Spread the samples across the clip. Consecutive frames from the
        // start only characterise whatever the opening shot happens to be
        // doing, which on a 240-frame clip is not representative.
        const SAMPLES: u32 = 10;
        let span = clip.frames.max(2) - 1;

        let mut worst = 0.0f32;
        let mut total_reduction = 0.0f32;
        for sample in 0..SAMPLES {
            let index = (sample as u64 * span / SAMPLES as u64) as u32;
            let prev = crate::media::frame_at(&clip, index as f64 * step, None).expect("prev");
            let cur =
                crate::media::frame_at(&clip, (index + 1) as f64 * step, None).expect("cur");

            let started = std::time::Instant::now();
            let motion = analyze(&prev, &cur).expect("analyze");
            let elapsed = started.elapsed();
            let flow = &motion.flow;
            let warped = warp(&prev, flow, 1.0);

            let before = mean_abs_diff(&prev, &cur);
            let after = mean_abs_diff(&warped, &cur);
            let reduction = if before > 0.0 {
                (1.0 - after / before) * 100.0
            } else {
                0.0
            };
            worst = worst.max(after / before.max(1e-6));
            total_reduction += reduction;

            println!(
                "pair {index}->{}: residual {before:.2} -> {after:.2} ({reduction:+.0}%) · \
                 analyze {:.0}ms · motion {:.2}px · confidence {:.2}{}",
                index + 1,
                elapsed.as_secs_f32() * 1000.0,
                flow.mean_magnitude(),
                motion.mean_confidence(),
                if motion.cut { " · CUT" } else { "" }
            );
        }

        println!(
            "mean residual reduction across {SAMPLES} pairs: {:+.1}%",
            total_reduction / SAMPLES as f32
        );

        // Warping must not make frames *less* similar. A stricter bar belongs
        // on the full lock, where the confidence gate can reject bad flow;
        // here the only claim is that the raw estimate is directionally right.
        assert!(
            worst < 1.0,
            "warping increased residual on at least one pair (ratio {worst:.2})"
        );
    }
}
