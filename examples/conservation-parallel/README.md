# Demo of Conservative Schemes, Parallel Version

This parallel version is running with the help of [Rayon](https://docs.rs/rayon/1.3.0/rayon/).

The comparison of the speed is recorded as the following.

```bash
cargo run --example conservation-parallel
157.10s user 52.97s system 282% cpu 1:14.32 total
```

```bash
cargo run --example conservation
100.33s user 29.12s system 117% cpu 1:50.23 total
```

## Usage

```bash
cargo run --example conservation-parallel
```

## Note

Please visit [here](https://yuanyuyuan.github.io/presentations/fdm).
