# Deploying to a free Hugging Face Space

A free **CPU basic** Space (2 vCPU, 16 GB RAM) can run the whole app, including
Ollama with `llama3.2:1b`. Expect answers to take 5–20 seconds.

> This deployment path has not been run end-to-end yet. If a step fails, the
> Space's **Logs** tab shows the build output.

## Steps

1. Create a free account at https://huggingface.co.
2. Create a new Space: **Docker** SDK, **Blank** template, **CPU basic** hardware.
3. In the Space's **Settings → Variables and secrets**, add a secret named
   `RAG_JWT_SECRET` with a long random value. Generate one with:

   ```bash
   python -c "import secrets; print(secrets.token_urlsafe(48))"
   ```

4. Clone the Space next to this project and copy the app into it:

   ```bash
   git clone https://huggingface.co/spaces/<your-username>/<space-name> space
   cp -r app evaluation scripts data frontend requirements.txt space/
   cp deploy/huggingface/Dockerfile space/Dockerfile
   ```

   Don't copy `frontend/node_modules`, `data/index` or `data/app.db`.

5. Put this header at the top of `space/README.md` (Spaces read it for settings):

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

6. Commit and push. The first build takes 10–20 minutes because it downloads
   PyTorch and the models:

   ```bash
   cd space
   git add .
   git commit -m "Deploy"
   git push
   ```

## Limitations of the free tier

- Free Spaces sleep after 48 hours without visitors and wake on the next visit.
- There is no persistent disk: accounts, chat history and uploaded documents are
  lost whenever the Space restarts. The sample documents are rebuilt from the image.
