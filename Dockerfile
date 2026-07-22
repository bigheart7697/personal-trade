# syntax=docker/dockerfile:1

# --- build stage ---
FROM golang:1.23 AS build
WORKDIR /src
COPY go.mod ./
COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -o /tradeforge ./cmd/tradeforge

# --- runtime stage ---
FROM gcr.io/distroless/static
COPY --from=build /tradeforge /tradeforge
EXPOSE 8420
ENTRYPOINT ["/tradeforge", "serve"]
