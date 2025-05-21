# Используем официальный образ Python
FROM python:3.12-alpine as builder

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /uvx /bin/
# COPY --from=ghcr.io/astral-sh/uv:0.5.11 /usr/local/bin/uv /usr/local/bin/uv
# Устанавливаем Poetry
ENV POETRY_VERSION=2.0.0

RUN pip install "poetry==$POETRY_VERSION"

# Копируем зависимости
COPY pyproject.toml ./

# Устанавливаем зависимости системы
RUN poetry config virtualenvs.in-project true && \
    poetry install --no-root --only main --no-interaction --no-ansi

# Финальный образ
FROM python:3.12-alpine

WORKDIR /app

# Копируем виртуальное окружение и исходный код
COPY --from=builder /app/.venv ./.venv
COPY . .

# Активируем виртуальное окружение
ENV PATH="/app/.venv/bin:$PATH"

RUN pip install uv
# Compile bytecode
# Ref: https://docs.astral.sh/uv/guides/integration/docker/#compiling-bytecode
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

RUN uv --version

# Install dependencies
# Ref: https://docs.astral.sh/uv/guides/integration/docker/#intermediate-layers
# RUN --mount=type=cache,target=/root/.cache/uv \
#     --mount=type=bind,source=uv.lock,target=uv.lock \
#     --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
#     uv sync --frozen --no-install-project

ENV PYTHONPATH="${PYTHONPATH}:/app"

# Копируем настройки
COPY ./settings/prod.env /app/settings/prod.env

COPY ./scripts /app/scripts

COPY ./pyproject.toml ./uv.lock ./alembic.ini /app/

COPY . /app

# Sync the project
# Ref: https://docs.astral.sh/uv/guides/integration/docker/#intermediate-layers
# RUN --mount=type=cache,target=/root/.cache/uv \
#     uv sync

EXPOSE 8000

CMD ["python", "run_app.py"]

# CMD ["fastapi", "run", "--workers", "4", "run_app.py"]

# # Команда запуска
# CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]