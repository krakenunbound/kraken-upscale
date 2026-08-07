//! Native file drag-and-drop.
//!
//! Slint has no cross-platform hook for OS file drops, so on Windows the
//! window procedure is subclassed to catch `WM_DROPFILES` directly. The
//! subclass runs on the UI thread (it *is* the message loop), so the callback
//! can touch the window without marshalling.

#[cfg(windows)]
mod imp {
    use std::cell::RefCell;
    use std::path::PathBuf;

    use raw_window_handle::{HasWindowHandle, RawWindowHandle};
    use windows_sys::Win32::Foundation::{HWND, LPARAM, LRESULT, WPARAM};
    use windows_sys::Win32::UI::Shell::{DragAcceptFiles, DragFinish, DragQueryFileW, HDROP};
    use windows_sys::Win32::UI::WindowsAndMessaging::{
        CallWindowProcW, SetWindowLongPtrW, GWLP_WNDPROC, WM_DROPFILES, WNDPROC,
    };

    type DropSink = Option<Box<dyn Fn(PathBuf)>>;

    thread_local! {
        static PREV_PROC: RefCell<isize> = const { RefCell::new(0) };
        static SINK: RefCell<DropSink> = const { RefCell::new(None) };
    }

    unsafe extern "system" fn wnd_proc(
        hwnd: HWND,
        msg: u32,
        wparam: WPARAM,
        lparam: LPARAM,
    ) -> LRESULT {
        if msg == WM_DROPFILES {
            let hdrop = wparam as HDROP;

            // Only the first file: this is a single-clip editor, and silently
            // loading an arbitrary one out of a multi-file drop is worse than
            // taking the one the user aimed with.
            let len = DragQueryFileW(hdrop, 0, std::ptr::null_mut(), 0);
            if len > 0 {
                let mut buf = vec![0u16; len as usize + 1];
                let got = DragQueryFileW(hdrop, 0, buf.as_mut_ptr(), buf.len() as u32);
                if got > 0 {
                    let path = PathBuf::from(String::from_utf16_lossy(&buf[..got as usize]));
                    SINK.with(|s| {
                        if let Some(f) = s.borrow().as_ref() {
                            f(path);
                        }
                    });
                }
            }
            DragFinish(hdrop);
            return 0;
        }

        let prev = PREV_PROC.with(|p| *p.borrow());
        CallWindowProcW(
            std::mem::transmute::<isize, WNDPROC>(prev),
            hwnd,
            msg,
            wparam,
            lparam,
        )
    }

    /// Registers `on_drop` for files dropped onto `window`.
    ///
    /// Must be called after the window is shown — there is no HWND before that.
    pub fn attach<F>(window: &slint::Window, on_drop: F) -> Result<(), String>
    where
        F: Fn(PathBuf) + 'static,
    {
        // Bound separately: the intermediate would otherwise be dropped
        // before the borrowed handle is read.
        let owner = window.window_handle();
        let handle = owner
            .window_handle()
            .map_err(|e| format!("no window handle: {e}"))?;

        let RawWindowHandle::Win32(w) = handle.as_raw() else {
            return Err("not a Win32 window".into());
        };
        let hwnd = w.hwnd.get() as HWND;

        SINK.with(|s| *s.borrow_mut() = Some(Box::new(on_drop)));

        unsafe {
            let prev = SetWindowLongPtrW(hwnd, GWLP_WNDPROC, wnd_proc as usize as isize);
            if prev == 0 {
                return Err("could not subclass window".into());
            }
            PREV_PROC.with(|p| *p.borrow_mut() = prev);
            DragAcceptFiles(hwnd, 1);
        }

        Ok(())
    }
}

#[cfg(not(windows))]
mod imp {
    use std::path::PathBuf;

    pub fn attach<F>(_: &slint::Window, _: F) -> Result<(), String>
    where
        F: Fn(PathBuf) + 'static,
    {
        Err("file drop is only implemented for Windows".into())
    }
}

pub use imp::attach;
