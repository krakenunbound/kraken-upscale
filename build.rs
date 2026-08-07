fn main() {
    slint_build::compile("ui/app.slint").expect("failed to compile Slint UI");

    // Embed the application icon and version metadata into the executable, so
    // Explorer, the taskbar and Alt-Tab show the kraken rather than the default
    // Windows placeholder. This is separate from the Slint `icon:` property,
    // which only sets the in-app window icon at runtime.
    #[cfg(windows)]
    {
        println!("cargo:rerun-if-changed=assets/kraken.ico");

        let mut res = winresource::WindowsResource::new();
        res.set_icon("assets/kraken.ico");
        res.set("ProductName", "Kraken Upscale");
        res.set("FileDescription", "Kraken Upscale — AI video upscaler");
        res.set("CompanyName", "Kraken");
        res.set("LegalCopyright", "");
        if let Err(e) = res.compile() {
            // A missing resource compiler should not stop a developer build;
            // the only loss is the icon.
            println!("cargo:warning=could not embed icon: {e}");
        }
    }
}
