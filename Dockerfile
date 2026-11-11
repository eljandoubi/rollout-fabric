FROM python:3.12-slim AS build
WORKDIR /src
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir build && python -m build --wheel --outdir /dist

FROM python:3.12-slim
# docker build --build-arg EXTRAS=ray -t rollout-fabric:ray .   adds the Ray backend
ARG EXTRAS=""
COPY --from=build /dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/rollout_fabric-*.whl${EXTRAS:+[$EXTRAS]} && rm /tmp/*.whl
RUN useradd --create-home --uid 10001 fabric
USER fabric
WORKDIR /home/fabric
ENTRYPOINT ["rollout-fabric"]
CMD ["compare", "--quick"]
