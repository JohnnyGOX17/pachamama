use log::info;
use pachamama::brokerage::alpaca::alpaca_client::{AccountType, AlpacaClient};
use std::env::set_var;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    unsafe { set_var("RUST_LOG", "INFO") };
    env_logger::init();

    info!("Initiating Pachamama...");

    let apca_client = AlpacaClient::new(AccountType::PAPER)?;
    apca_client.get_account_details().await
}
