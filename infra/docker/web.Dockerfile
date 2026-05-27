FROM node:20-alpine AS deps
WORKDIR /app
COPY apps/web/package.json apps/web/package.json
RUN corepack enable && cd apps/web && npm install --legacy-peer-deps

FROM node:20-alpine AS builder
WORKDIR /app
COPY --from=deps /app/apps/web/node_modules /app/apps/web/node_modules
COPY apps/web /app/apps/web
RUN cd apps/web && npm run build

FROM node:20-alpine AS runner
WORKDIR /app/apps/web
ENV NODE_ENV=production PORT=3000
COPY --from=builder /app/apps/web ./
EXPOSE 3000
CMD ["npm", "run", "start"]
