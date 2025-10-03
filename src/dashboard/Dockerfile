# Stage 1: Build React app
FROM node:18 AS build
WORKDIR /app

# Install dependencies
COPY package*.json ./
RUN npm install

# Copy all frontend source
COPY . .

# Build React/Vite app (output will be in /app/dist)
RUN npm run build

# Stage 2: Serve with Nginx
FROM nginx:1.25-alpine

# Copy build output
COPY --from=build /app/dist /usr/share/nginx/html

# Copy custom nginx config
COPY nginx.conf /etc/nginx/conf.d/default.conf
