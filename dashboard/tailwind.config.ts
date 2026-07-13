import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // Semantic risk palette — reused by tables and charts so the whole
        // dashboard reads as one system (accessible in light and dark).
        risk: {
          low: "#16a34a",
          medium: "#d97706",
          high: "#dc2626",
        },
        brand: "#4f46e5",
      },
    },
  },
  plugins: [],
};

export default config;
