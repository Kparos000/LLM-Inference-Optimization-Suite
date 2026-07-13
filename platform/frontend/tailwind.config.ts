import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        graphite: {
          950: "#070b11",
          900: "#0b111a",
          850: "#101825",
          800: "#141f2f",
          700: "#1e2b40"
        },
        signal: {
          cyan: "#47d7ff",
          green: "#54e6a5",
          amber: "#ffbf5f",
          red: "#ff6b7a"
        }
      },
      boxShadow: {
        mission: "0 24px 80px rgba(0, 0, 0, 0.35)"
      }
    }
  },
  plugins: []
};

export default config;

