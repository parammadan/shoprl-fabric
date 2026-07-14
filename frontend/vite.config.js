import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
// Dev proxy: browser talks only to the Vite origin; /api/* -> FastAPI :8000.
// Keeps the backend UNCHANGED (no CORS edit).
export default defineConfig({
    plugins: [react()],
    server: {
        port: 5173,
        proxy: {
            "/api": {
                target: process.env.SHOPRL_API || "http://127.0.0.1:8000",
                changeOrigin: true,
                rewrite: function (p) { return p.replace(/^\/api/, ""); },
            },
        },
    },
});
