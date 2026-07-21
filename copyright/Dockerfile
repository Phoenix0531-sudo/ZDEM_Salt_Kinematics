FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget ca-certificates libicu76 \
    && rm -rf /var/lib/apt/lists/*

# .NET SDK 8.0
RUN wget -q https://dot.net/v1/dotnet-install.sh -O /tmp/dotnet-install.sh \
    && chmod +x /tmp/dotnet-install.sh \
    && /tmp/dotnet-install.sh --channel 8.0 --install-dir /usr/share/dotnet \
    && ln -s /usr/share/dotnet/dotnet /usr/bin/dotnet

# Python deps
RUN pip install --no-cache-dir python-docx

ENV DOTNET_CLI_UI_LANGUAGE=en
ENV PATH="/usr/share/dotnet:${PATH}"

WORKDIR /work
