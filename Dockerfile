FROM python:3.12-slim-bookworm
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy the application into the container.
COPY . /app

# Install the application dependencies.
WORKDIR /app
RUN uv sync --frozen --no-cache

RUN cd web_app

RUN set FLASK_APP = app.py



# Run the application.
CMD ["uv","run", "flask", "run", "--port", "80", "--host", "0.0.0.0"]
