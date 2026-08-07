//! Per-stage timing for the export pipeline.
//!
//! Phase 0 of ENHANCEMENTS.md: before the temporal lock lands there has to be a
//! baseline showing where the time actually goes, otherwise "the lock made it
//! slower" is unfalsifiable and every later optimisation is guesswork.
//!
//! Fields are `Cell`s so the accumulator can be shared by `&` between the
//! decode and process closures in `export::run`, which cannot both hold a
//! mutable borrow of the same value.

use std::cell::Cell;
use std::time::{Duration, Instant};

#[derive(Default)]
pub struct Stages {
    frames: Cell<u64>,
    decode: Cell<Duration>,
    interpolate: Cell<Duration>,
    model: Cell<Duration>,
    lock: Cell<Duration>,
    post: Cell<Duration>,
    encode: Cell<Duration>,
    /// How much of the frame the lock actually held, averaged over locked
    /// frames. Without this the log says the lock ran but not whether it did
    /// anything — a run where confidence collapsed everywhere looks identical
    /// to one where it worked.
    confidence_sum: Cell<f32>,
    locked_frames: Cell<u64>,
    /// The first model call separately, because it carries one-off costs —
    /// CUDA session creation, cuDNN algorithm autotuning, memory arena growth
    /// — that never recur. Folding it into the average makes short runs look
    /// far slower than steady state and makes A/B comparisons meaningless.
    model_first: Cell<Duration>,
}

/// Times `f`, adding the elapsed time to `slot`.
fn accumulate<T>(slot: &Cell<Duration>, f: impl FnOnce() -> T) -> T {
    let start = Instant::now();
    let out = f();
    slot.set(slot.get().saturating_add(start.elapsed()));
    out
}

impl Stages {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn frame_done(&self) {
        self.frames.set(self.frames.get() + 1);
    }

    pub fn decode<T>(&self, f: impl FnOnce() -> T) -> T {
        accumulate(&self.decode, f)
    }

    pub fn interpolate<T>(&self, f: impl FnOnce() -> T) -> T {
        accumulate(&self.interpolate, f)
    }

    pub fn model<T>(&self, f: impl FnOnce() -> T) -> T {
        let before = self.model.get();
        let out = accumulate(&self.model, f);
        if self.frames.get() == 0 {
            self.model_first.set(self.model.get().saturating_sub(before));
        }
        out
    }

    pub fn lock<T>(&self, f: impl FnOnce() -> T) -> T {
        accumulate(&self.lock, f)
    }

    pub fn record_confidence(&self, mean: f32) {
        self.confidence_sum.set(self.confidence_sum.get() + mean);
        self.locked_frames.set(self.locked_frames.get() + 1);
    }

    pub fn post<T>(&self, f: impl FnOnce() -> T) -> T {
        accumulate(&self.post, f)
    }

    pub fn encode<T>(&self, f: impl FnOnce() -> T) -> T {
        accumulate(&self.encode, f)
    }

    /// Everything the stages measured, which is less than wall clock — process
    /// startup, pipe backpressure and the final encoder flush sit outside.
    pub fn measured(&self) -> Duration {
        self.decode
            .get()
            .saturating_add(self.interpolate.get())
            .saturating_add(self.model.get())
            .saturating_add(self.lock.get())
            .saturating_add(self.post.get())
            .saturating_add(self.encode.get())
    }

    /// One-line breakdown for the activity log, in per-frame milliseconds with
    /// the share of measured time. Per-frame is the number that matters when
    /// comparing runs of different lengths.
    pub fn summary(&self, wall: Duration) -> String {
        let frames = self.frames.get().max(1);
        let per_frame = |d: Duration| d.as_secs_f64() * 1000.0 / frames as f64;
        let measured = self.measured().as_secs_f64().max(1e-9);
        let share = |d: Duration| d.as_secs_f64() / measured * 100.0;

        let mut parts = vec![format!(
            "decode {:.0}ms/{:.0}%",
            per_frame(self.decode.get()),
            share(self.decode.get())
        )];

        // Only present when interpolation actually ran.
        if !self.interpolate.get().is_zero() {
            parts.push(format!(
                "rife {:.0}ms/{:.0}%",
                per_frame(self.interpolate.get()),
                share(self.interpolate.get())
            ));
        }

        parts.push(format!(
            "model {:.0}ms/{:.0}%",
            per_frame(self.model.get()),
            share(self.model.get())
        ));

        // Only present when the temporal lock actually ran.
        if !self.lock.get().is_zero() {
            let locked = self.locked_frames.get().max(1);
            parts.push(format!(
                "lock {:.0}ms/{:.0}% (held {:.2} over {} frames)",
                per_frame(self.lock.get()),
                share(self.lock.get()),
                self.confidence_sum.get() / locked as f32,
                self.locked_frames.get()
            ));
        }

        parts.push(format!(
            "post {:.0}ms/{:.0}%",
            per_frame(self.post.get()),
            share(self.post.get())
        ));
        parts.push(format!(
            "encode {:.0}ms/{:.0}%",
            per_frame(self.encode.get()),
            share(self.encode.get())
        ));

        // Steady state excludes the first frame's one-off GPU warm-up, which
        // is the number that actually predicts a long export.
        let warmup = self.model_first.get();
        let steady = if self.frames.get() > 1 {
            let rest = self.model.get().saturating_sub(warmup);
            format!(
                " · model steady {:.0}ms (warm-up {:.0}ms excluded)",
                rest.as_secs_f64() * 1000.0 / (self.frames.get() - 1) as f64,
                warmup.as_secs_f64() * 1000.0
            )
        } else {
            String::new()
        };

        format!(
            "stage timings · {} frames · {:.2}s/frame wall · {} · unaccounted {:.0}ms/frame{}",
            self.frames.get(),
            wall.as_secs_f64() / frames as f64,
            parts.join(" · "),
            per_frame(wall.saturating_sub(self.measured())),
            steady
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accumulates_per_stage_and_counts_frames() {
        let s = Stages::new();
        s.model(|| std::thread::sleep(Duration::from_millis(12)));
        s.post(|| std::thread::sleep(Duration::from_millis(4)));
        s.frame_done();

        assert!(s.measured() >= Duration::from_millis(15));

        let text = s.summary(Duration::from_millis(40));
        assert!(text.contains("1 frames"), "{text}");
        assert!(text.contains("model"), "{text}");
        assert!(text.contains("unaccounted"), "{text}");
        // Interpolation never ran, so it stays out of the line.
        assert!(!text.contains("rife"), "{text}");
    }

    #[test]
    fn separates_first_frame_warmup_from_steady_state() {
        let s = Stages::new();
        // Frame 1 is slow, as a cold CUDA session is.
        s.model(|| std::thread::sleep(Duration::from_millis(60)));
        s.frame_done();
        // Frames 2 and 3 are representative.
        for _ in 0..2 {
            s.model(|| std::thread::sleep(Duration::from_millis(10)));
            s.frame_done();
        }

        let text = s.summary(Duration::from_millis(90));
        assert!(text.contains("model steady"), "{text}");
        assert!(text.contains("warm-up"), "{text}");

        // Steady state must reflect frames 2-3 (~10ms), not the 27ms average
        // that including the cold frame would produce.
        let steady: f64 = text
            .split("model steady ")
            .nth(1)
            .and_then(|t| t.split("ms").next())
            .and_then(|t| t.parse().ok())
            .expect("steady figure present");
        assert!((5.0..20.0).contains(&steady), "steady was {steady}ms: {text}");
    }

    #[test]
    fn reports_interpolation_only_when_used() {
        let s = Stages::new();
        s.interpolate(|| std::thread::sleep(Duration::from_millis(2)));
        s.frame_done();
        assert!(s.summary(Duration::from_millis(10)).contains("rife"));
    }
}
