FROM python:3.12-slim-bookworm
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy the application into the container.
COPY . /app

# Install the application dependencies.
WORKDIR /app
RUN uv sync --frozen --no-cache

# Install gunicorn
RUN uv pip install gunicorn

# Set environment variables
ENV FLASK_APP=web_app/app.py

# Run the application with Gunicorn
CMD ["uv", "run", "gunicorn", "-c", "gunicorn.conf.py", "web_app.app:app"]
