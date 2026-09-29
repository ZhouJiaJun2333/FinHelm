import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// 开发时 npm run dev（5173），/api 转给 python run_web.py 起的后端
export default defineConfig({
  plugins: [react()],
  server: { proxy: { "/api": "http://127.0.0.1:8765" } },
});
