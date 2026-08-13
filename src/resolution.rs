//! Exact output-resolution goals shared by video and still-image pipelines.
//!
//! Named goals target the short edge: height for landscape, width for portrait.
//! This gives familiar 720p/1080p/1440p/2160p sizing without cropping or
//! stretching vertical sources. Multiplier goals remain available for unusual
//! aspect ratios and comparison work.

pub const GOALS: [u32; 7] = [720, 1080, 1440, 2160, 2, 3, 4];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum GoalClass {
    Downscale = 0,
    NearSource = 1,
    Upscale = 2,
    LargeUpscale = 3,
}

/// Classifies a goal for source-aware UI coloring.
///
/// Named goals are compared with the source short edge because that is the
/// edge they target. A five-percent band reads as "essentially the same" and
/// avoids marking a 1078px source versus 1080p as a meaningful upscale.
pub fn classify(width: u32, height: u32, goal: u32) -> GoalClass {
    if goal <= 4 {
        return if goal >= 3 {
            GoalClass::LargeUpscale
        } else {
            GoalClass::Upscale
        };
    }
    let short_edge = width.min(height);
    if short_edge == 0 {
        return GoalClass::NearSource;
    }
    let target = goal as u64 * 100;
    let source = short_edge as u64;
    if target < source * 95 {
        GoalClass::Downscale
    } else if target <= source * 105 {
        GoalClass::NearSource
    } else if target < source * 300 {
        GoalClass::Upscale
    } else {
        GoalClass::LargeUpscale
    }
}

pub fn from_index(index: i32) -> u32 {
    GOALS[(index.max(0) as usize).min(GOALS.len() - 1)]
}

pub fn label(goal: u32) -> String {
    match goal {
        720 => "HD-720p".into(),
        1080 => "FHD-1080p".into(),
        1440 => "2K-1440p".into(),
        2160 => "4K-2160p".into(),
        scale => format!("{scale}x"),
    }
}

pub fn display_label(goal: u32) -> String {
    match goal {
        720 => "HD · 720p".into(),
        1080 => "Full HD · 1080p".into(),
        1440 => "2K · 1440p".into(),
        2160 => "4K / UHD · 2160p".into(),
        scale => format!("Multiplier · x{scale}"),
    }
}

/// Returns exact aspect-preserving output dimensions.
pub fn dimensions(width: u32, height: u32, goal: u32, even: bool) -> (u32, u32) {
    if goal <= 4 {
        return (
            adjust(width.saturating_mul(goal), even),
            adjust(height.saturating_mul(goal), even),
        );
    }
    if width == 0 || height == 0 {
        return (0, 0);
    }

    let (out_width, out_height) = if width >= height {
        let out_height = goal;
        let out_width = ((width as u64 * goal as u64 + height as u64 / 2) / height as u64)
            .min(u32::MAX as u64) as u32;
        (out_width, out_height)
    } else {
        let out_width = goal;
        let out_height = ((height as u64 * goal as u64 + width as u64 / 2) / width as u64)
            .min(u32::MAX as u64) as u32;
        (out_width, out_height)
    };
    (adjust(out_width, even), adjust(out_height, even))
}

fn adjust(value: u32, even: bool) -> u32 {
    if even && !value.is_multiple_of(2) {
        value.saturating_add(1)
    } else {
        value
    }
}

/// Integer working scale needed before the exact final resize.
pub fn working_scale(width: u32, height: u32, goal: u32, even: bool) -> Result<u32, String> {
    let (out_width, out_height) = dimensions(width, height, goal, even);
    if out_width <= width && out_height <= height {
        return Err(format!(
            "{} is not larger than the {}×{} source; choose a higher resolution goal",
            display_label(goal),
            width,
            height
        ));
    }
    let ceil_ratio = |out: u32, source: u32| out.saturating_add(source - 1) / source;
    let scale = ceil_ratio(out_width, width)
        .max(ceil_ratio(out_height, height))
        .max(2);
    if scale > 4 {
        return Err(format!(
            "{} needs about {scale}x enlargement from {}×{}; the current pipeline supports up to 4x",
            display_label(goal),
            width,
            height
        ));
    }
    Ok(scale)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn named_goals_follow_orientation_and_preserve_aspect() {
        assert_eq!(dimensions(1280, 720, 1080, true), (1920, 1080));
        assert_eq!(dimensions(720, 1280, 1080, true), (1080, 1920));
        assert_eq!(dimensions(736, 1280, 1080, true), (1080, 1878));
    }

    #[test]
    fn multiplier_goals_still_work() {
        assert_eq!(dimensions(704, 1248, 2, true), (1408, 2496));
        assert_eq!(working_scale(704, 1248, 2, true), Ok(2));
    }

    #[test]
    fn rejects_downscale_and_more_than_four_x() {
        assert!(working_scale(1920, 1080, 720, true).is_err());
        assert!(working_scale(320, 180, 2160, true).is_err());
    }

    #[test]
    fn classifies_goals_relative_to_the_source_short_edge() {
        assert_eq!(classify(3840, 2160, 1080), GoalClass::Downscale);
        assert_eq!(classify(1920, 1080, 1080), GoalClass::NearSource);
        assert_eq!(classify(1920, 1078, 1080), GoalClass::NearSource);
        assert_eq!(classify(1280, 720, 1080), GoalClass::Upscale);
        assert_eq!(classify(640, 360, 1440), GoalClass::LargeUpscale);
        assert_eq!(classify(8000, 8000, 2), GoalClass::Upscale);
        assert_eq!(classify(8000, 8000, 4), GoalClass::LargeUpscale);
    }
}
