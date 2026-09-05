FROM node:20-alpine AS builder
WORKDIR /app
COPY package.json .
RUN npm install
COPY . .
# Vite only exposes env vars prefixed VITE_ to import.meta.env — the old
# REACT_APP_ names were a CRA leftover and were never actually read by the
# app (see src/index.jsx, which reads import.meta.env.VITE_GMAPS_API_KEY).
ARG VITE_GMAPS_API_KEY=""
ENV VITE_GMAPS_API_KEY=$VITE_GMAPS_API_KEY
RUN npm run build

FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
