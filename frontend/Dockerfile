FROM node:20-alpine AS builder
WORKDIR /app
COPY package.json .
RUN npm install
COPY . .
ARG REACT_APP_API_URL=http://localhost:5000/api
ARG REACT_APP_GMAPS_API_KEY=""
ENV REACT_APP_API_URL=$REACT_APP_API_URL
ENV REACT_APP_GMAPS_API_KEY=$REACT_APP_GMAPS_API_KEY
RUN npm run build

FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
