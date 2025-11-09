# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Pachamama is an algorithmic trading application built in Rust that performs fundamental analysis on the Alpaca Markets trading platform. The project is in early development and currently implements basic account information retrieval from the Alpaca paper trading API.

## Build and Development Commands

### Building
```bash
cargo build          # Build in debug mode
cargo build --release # Build optimized release binary
```

### Running
```bash
cargo run            # Run the application
RUST_LOG=info cargo run  # Run with logging enabled
```

### Testing
```bash
cargo test           # Run all tests
cargo test <test_name> # Run a specific test
```

### Other Commands
```bash
cargo check          # Quick compile check without building
cargo fmt            # Format code
cargo clippy         # Run linter
```

## Architecture

### Module Structure

- `src/main.rs`: Entry point that demonstrates basic Alpaca API account retrieval
- `src/lib.rs`: Library root exposing the `api` module
- `src/api/`: Alpaca API client implementation
  - `mod.rs`: API module root
  - `account.rs`: Account-related API types and responses

### API Integration

The codebase uses the blocking reqwest client to interact with Alpaca's REST API. API calls require authentication via environment variables:

- `APCA_API_KEY_ID`: Alpaca API key ID
- `APCA_API_SECRET_KEY`: Alpaca API secret key

Authentication is done via HTTP headers (`APCA-API-KEY-ID` and `APCA-API-SECRET-KEY`) rather than OAuth.

### Current API Endpoints

- **Account**: `GET https://paper-api.alpaca.markets/v2/account` - Retrieves account details
  - Implemented in `src/api/account.rs` as `GetAccountResp`
  - Uses strongly-typed deserialization with serde

### Design Patterns

- **Type-safe API responses**: All API responses are modeled as Rust structs with serde serialization/deserialization
- **Error handling**: Uses `Result<T, Box<dyn std::error::Error>>` for recoverable errors
- **Logging**: Uses the `log` crate with `env_logger` for runtime logging control

## Key Dependencies

- `reqwest`: HTTP client (blocking mode enabled)
- `serde`: JSON serialization/deserialization
- `tokio`: Async runtime (full features enabled, though not yet utilized)
- `chrono`: Date/time handling for API timestamps
- `log` + `env_logger`: Logging infrastructure

## Resources

- [Alpaca Markets API Documentation](https://docs.alpaca.markets/)
- [Alpaca Paper Trading Dashboard](https://app.alpaca.markets/paper/dashboard/overview)
- The project uses paper trading endpoints for development and testing
