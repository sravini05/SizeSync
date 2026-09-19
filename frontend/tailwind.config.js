/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        mist: "#F5F6F3",
        ink: "#12161B",
        thread: "#E14B36",
        brass: "#AD8A54",
        sage: "#3C8B6C",
      },
      fontFamily: {
        display: ["Fraunces", "serif"],
        body: ["Inter", "sans-serif"],
        mono: ["'IBM Plex Mono'", "monospace"],
      },
      backgroundImage: {
        "grid-fine":
          "linear-gradient(to right, rgba(18,22,27,0.05) 1px, transparent 1px), linear-gradient(to bottom, rgba(18,22,27,0.05) 1px, transparent 1px)",
      },
      backgroundSize: {
        "grid-fine": "24px 24px",
      },
    },
  },
  plugins: [],
};
