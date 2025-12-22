# Training Monitoring Documentation

The `hafs training` CLI provides tools for monitoring active and historical training campaigns.

## Basic Monitoring

### `hafs training status`
Shows the current health of the training system, including:
- **Campaign Progress**: Running status, PID, sample progress, and generation rate.
- **System Resources**: CPU, Memory, and Disk usage with color-coded health indicators.
- **Services**: Embedding service and knowledge base status.
- **Remote Inference**: Status of remote nodes like `medical-mechanica`.
- **Issues**: Automatic detection of stalled campaigns, low quality rates, or resource pressure.

Options:
- `--watch`, `-w`: Continuous updates.
- `--interval`, `-i`: Update frequency (default 30s).

## Historical Exploration

### `hafs training history`
Lists all historical training campaigns found in the datasets directory.
- Shows Run ID, Creation Time, Domains, Sample Count, Quality Score, and Duration.

### `hafs training show [RUN_ID]`
Provides detailed information for a specific historical run:
- Metadata: Template used, creation date, target vs final samples.
- Performance: Average quality score and duration.
- Distribution: Domain-specific sample counts.
- Artifacts: Map to the generated `.jsonl` files and the associated log file.

### `hafs training logs [RUN_ID]`
Displays logs for the current or a historical campaign.
- If `RUN_ID` is provided, it attempts to find the log file matching that campaign's timestamp.
- Options: `--follow` (`-f`), `--lines` (`-n`).

## Remote Node Support

The monitoring system is aware of the `medical-mechanica` node (and other remote nodes configured in `nodes.toml`). It specifically reports on:
- Connectivity status (Online/Offline).
- GPU hardware and memory availability.
- Inference model inventory.
