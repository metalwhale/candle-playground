# candle-playground
Burn my heart, not my computer

## Local development
1. Change to [`deployment-local`](./deployment-local/) directory:
    ```bash
    cd ./deployment-local/
    ```
2. Start and get inside the container:
    ```bash
    docker compose up --build --remove-orphans -d
    docker compose exec candle-playground bash
    ```
3. Run example:
    ```bash
    cargo run --example qwen --release -- --prompt "Hello darkness, my old friend"
    ```
