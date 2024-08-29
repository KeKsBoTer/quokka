bash build_wasm.sh &&\
cargo build --release --bin quokka --features colormaps &&\
cargo build --release --bin quokka &&\
maturin develop