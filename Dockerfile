FROM python:3.12-slim-bookworm
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy the application into the container.
COPY . /app

# Install the application dependencies.
WORKDIR /app
RUN uv sync --frozen --no-cache

RUN set FLASK_APP=web_app/app.py

RUN cd web_app

# Run the application.
CMD ["/app/.venv/bin/flask", "run", "--port", "80", "--host", "0.0.0.0"]
