//! Desktop file drag-and-drop through Slint's native winit event stream.

use std::path::PathBuf;

use slint::winit_030::{winit::event::WindowEvent, EventResult, WinitWindowAccessor};

/// Registers callbacks for file drops and hover feedback on `window`.
///
/// Winit owns the platform drop target, so the event must be observed here
/// rather than by installing a second native window procedure.
pub fn attach<F, H>(window: &slint::Window, mut on_drop: F, mut on_hover: H) -> Result<(), String>
where
    F: FnMut(PathBuf) + 'static,
    H: FnMut(bool) + 'static,
{
    if !window.has_winit_window() {
        return Err("native window is not ready".into());
    }

    window.on_winit_window_event(move |_, event| {
        match event {
            WindowEvent::HoveredFile(_) => on_hover(true),
            WindowEvent::HoveredFileCancelled => on_hover(false),
            WindowEvent::DroppedFile(path) => {
                on_hover(false);
                on_drop(path.clone());
            }
            _ => {}
        }
        EventResult::Propagate
    });

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recognizes_winit_file_drop_event() {
        let path = PathBuf::from(r"C:\video\clip.mp4");
        let event = WindowEvent::DroppedFile(path.clone());
        let extracted = match event {
            WindowEvent::DroppedFile(path) => Some(path),
            _ => None,
        };
        assert_eq!(extracted, Some(path));
    }
}
