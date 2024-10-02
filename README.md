# candle-playground
Burn my heart, not my computer

## Local development
1. Change to [`deployment-local`](./deployment-local/) directory:
    ```bash
    cd ./deployment-local/
    ```
2. Start the containers:
    ```bash
    docker compose up --build --remove-orphans -d
    ```

### Evaluation
1. Get inside `evaluation` container:
    ```bash
    docker compose exec evaluation bash
    ```
2. Install packages:
    ```bash
    poetry install --no-root
    ```
    This command will create a virtual environment in `../cache/poetry/virtualenvs`.
3. Activate the virtual environment:
    <pre>
    source ../cache/poetry/virtualenvs/<b>ENVIRONMENT_NAME</b>/bin/activate
    </pre>
4. Run inference:
    <pre>
    python main.py <b>IMAGE_PATH</b>
    </pre>

### Run examples
1. Get inside `candle-playground` container:
    ```bash
    docker compose exec candle-playground bash
    ```
2. Run the corresponding command for each example:
    <pre>
    cargo run --example qwen --release -- --prompt <b>PROMPT</b>
    cargo run --example got_ocr --release -- --image-path <b>IMAGE_PATH</b>
    cargo run --example houou --release -- --prompt <b>PROMPT</b>
    </pre>
