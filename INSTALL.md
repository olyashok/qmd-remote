# QMD Remote Installation Guide

Quick setup guide for installing QMD with remote LLM backend on a new server.

## Prerequisites

- Linux or macOS
- Network access to the QMD model server (192.168.5.163)

## Installation

### 1. Install Bun (JavaScript runtime)

```bash
curl -fsSL https://bun.sh/install | bash
source ~/.bashrc  # or source ~/.zshrc on macOS
```

### 2. Clone and Install QMD

```bash
# Clone the repository to home folder
cd ~
git clone https://github.com/olyashok/qmd-remote.git
cd qmd-remote

# Install dependencies
bun install

# Make qmd executable
chmod +x qmd

# Add to PATH (choose one method)

# Option A: Symlink to /usr/local/bin (requires sudo)
sudo ln -sf ~/qmd-remote/qmd /usr/local/bin/qmd

# Option B: Add to ~/.bashrc or ~/.zshrc (no sudo, recommended)
echo 'export PATH="$PATH:$HOME/qmd-remote"' >> ~/.bashrc
source ~/.bashrc
```

### 3. Configure Remote Endpoints

Set the remote model server endpoints:

```bash
qmd remote set \
  http://192.168.5.163:8081 \
  http://192.168.5.163:8082 \
  http://192.168.5.163:8083

# Verify configuration
qmd remote status
```

**Using LiteLLM for query expansion (generate):** Point the generate endpoint at your LiteLLM proxy and set the model name:

```bash
# Point generate at LiteLLM (keep embed/rerank on llama.cpp servers)
# Use the server IP, not localhost
qmd remote set \
  http://192.168.5.163:8081 \
  http://192.168.5.163:8082 \
  http://192.168.5.163:4000

# Set the fast model for expansion (required for LiteLLM)
qmd remote set --generate-model fast
# or: gpt-4o-mini, ollama/llama3, etc.
```

### 4. Configure Shared Data Directory (if available)

If you have access to the shared NFS mount:

```bash
# Check if the shared directory exists
if [ -d "/mnt/pve/nfs-cellect-artifacts/shape" ]; then
  # Set it as the persistent qmd directory
  qmd init /mnt/pve/nfs-cellect-artifacts/shape
  echo "✓ Configured to use shared .qmd index"
else
  echo "Shared directory not available - will use local index"
fi

# Verify configuration
qmd where
```

## Usage

### Basic Commands

```bash
# Search across all collections
qmd search "redevelopment plan requirements"

# Vector search (semantic)
qmd vsearch "parking regulations"

# Advanced query (expansion + vector + reranking)
qmd query "jersey city zoning rules" -n 5

# Search within a specific collection
qmd search "affordable housing" -c jersey_city_resolutions

# List available collections
qmd collection list

# Show index status
qmd status
```

### Working with Local Data

If you want to index your own markdown files:

```bash
# Create a portable index in your data directory
cd /path/to/your/data
qmd init .

# Add a collection
qmd collection add . --name my_docs --mask "**/*.md"

# Create embeddings (uses remote GPU)
qmd embed

# Search your data
qmd query "your search term"
```

### Collection Examples

Based on the current index, you can query:

- `jersey_city_code` - Municipal code documents
- `jersey_city_meeting_documents` - Planning board agendas and minutes
- `jersey_city_redevelopment_plans` - Redevelopment plans
- `jersey_city_resolutions` - City council resolutions
- `jersey_city_transcripts` - Meeting transcripts
- `nj_licenses`, `nj_parcels`, `nj_permits` - NJ state data

```bash
# Example: Find all resolutions about affordable housing
qmd search "affordable housing" -c jersey_city_resolutions -n 10

# Example: Find redevelopment plans with parking requirements
qmd query "parking requirements" -c jersey_city_redevelopment_plans
```

## Configuration Priority

QMD resolves the index location with the following priority:

1. **CLI flag:** `--qmd-dir /path/to/.qmd` (highest priority)
2. **Saved config:** Set via `qmd init /path` (persistent)
3. **Auto-discover:** Search upward from current directory

```bash
# One-time override
qmd --qmd-dir /other/path status

# Persistent config (works from anywhere)
qmd init /path/to/data

# Clear saved path (use auto-discovery)
qmd init clear

# Check current resolution
qmd where
```

## Troubleshooting

### Remote endpoints unreachable

```bash
# Check network connectivity
curl http://192.168.5.163:8081/health

# Verify remote config
qmd remote status

# Fall back to local models temporarily
qmd --local query "test"
```

### No collections found

```bash
# If on shared mount, make sure it's configured
qmd init /mnt/pve/nfs-cellect-artifacts/shape

# Otherwise, verify you're in the right directory
cd /path/to/data
qmd collection list
```

### Models not working

The remote server handles all model loading. If you see errors:

1. Check that the model server is running: `curl http://192.168.5.163:8081/health`
2. Contact the server administrator
3. Temporarily use local models: `qmd remote clear`

## Server Administration

If you're setting up the **model server** (not the client):

See the main repository documentation for:
- GPU passthrough configuration
- Docker Compose setup in `server/docker-compose.yml`
- Model downloads to `/mnt/pve/nfs-cellect-artifacts/models/`

The remote setup requires:
- NVIDIA GPU with CUDA support
- Docker with NVIDIA Container Toolkit
- ~3GB GPU VRAM for all three models
