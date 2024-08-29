use std::env;

use quokka::viewer;

#[pollster::main]
async fn main() -> anyhow::Result<()> {
    viewer(env::args()).await
}
