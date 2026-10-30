# Deploying to a Hugging Face Space

`deploy/huggingface/Dockerfile` packs the whole app into one container (API, website,
Ollama with `llama3.2:1b`), because a Space runs a single container. On **CPU basic**
hardware (2 vCPU, 16 GB RAM) expect answers to take 5–20 seconds.

> **Cost:** as of October 2026, creating a Docker Space needs a Hugging Face **PRO**
> account (the API answers `402 Payment Required` otherwise), even on CPU basic
> hardware. This guide has not been run end to end for that reason.

## Steps

1. With a PRO account, install the Hugging Face CLI and log in:

   ```bash
   pip install -U huggingface_hub
   hf auth login
   ```

2. Create the Space. Replace `<user>` with your username:

   ```bash
   hf repos create <user>/grounded-ai-knowledge-assistant --type space --sdk docker --public
   ```

3. In the Space's **Settings → Variables and secrets**, add a secret named
   `RAG_JWT_SECRET` with a long random value:

   ```bash
   python -c "import secrets; print(secrets.token_urlsafe(48))"
   ```

4. Build the upload folder from committed files only, so no local index, database or
   `node_modules` is included:

   ```bash
   mkdir space
   git archive HEAD app evaluation data/sample_docs frontend requirements.txt | tar -x -C space
   cp deploy/huggingface/Dockerfile space/Dockerfile
   ```

5. Create `space/README.md` with the Space settings header:

   ```yaml
   ---
   title: Grounded AI Knowledge Assistant
   emoji: 📚
   colorFrom: indigo
   colorTo: blue
   sdk: docker
   app_port: 7860
   ---
   ```

6. Upload it. The first build takes 10–20 minutes because it downloads PyTorch and the
   models; the Space's **Logs** tab shows the build output:

   ```bash
   hf upload <user>/grounded-ai-knowledge-assistant space . --repo-type space
   ```

## Limitations

- Spaces sleep after 48 hours without visitors and wake on the next visit.
- There is no persistent disk: accounts, chat history and uploaded documents are lost
  whenever the Space restarts. The sample documents are rebuilt from the image.
- The 1B answer model is weaker than the 3B model the benchmark was run with.
