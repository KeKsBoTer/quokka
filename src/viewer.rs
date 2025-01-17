use clap::Parser;
use std::{ffi::OsString, fmt::Debug, fs::File, io::BufReader, path::PathBuf};

use winit::{dpi::LogicalSize, window::WindowBuilder};

use crate::{open_window, presets::Preset, volume::Volume, RenderConfig};

#[derive(Debug, Parser)]
#[command(author, version, about)]
struct Opt {
    /// Input file
    input: PathBuf,

    #[arg(long, default_value_t = false)]
    no_vsync: bool,

    #[cfg(not(feature = "colormaps"))]
    colormap: PathBuf,

    preset: Option<PathBuf>,
}

pub async fn viewer<I, T>(args: I) -> anyhow::Result<()>
where
    I: IntoIterator<Item = T>,
    T: Into<OsString> + Clone,
{
    env_logger::init();
    let opt = Opt::try_parse_from(args)?;

    let data_file = File::open(&opt.input)?;

    let window_builder = WindowBuilder::new().with_inner_size(LogicalSize::new(800, 600));

    let volumes = Volume::load(BufReader::new(data_file)).expect("Failed to load volume");

    let preset = opt
        .preset
        .as_ref()
        .map(|path| {
            let reader = File::open(path)?;
            Preset::from_json(reader)
        })
        .transpose()
        .map_err(|e| anyhow::anyhow!("Failed to load preset: {}", e))?;

    open_window(
        window_builder,
        volumes,
        RenderConfig {
            no_vsync: opt.no_vsync,
            show_colormap_editor: true,
            show_volume_info: true,
            #[cfg(feature = "colormaps")]
            show_cmap_select: true,
            duration: None,
            preset,
        },
    )
    .await;
    Ok(())
}
