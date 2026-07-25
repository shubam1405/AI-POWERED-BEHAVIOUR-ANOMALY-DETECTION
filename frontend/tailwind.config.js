/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        dark: {
          bg: '#080c14',
          card: '#0e1526',
          border: '#1e293b',
          accent: '#1f293d',
        }
      }
    },
  },
  plugins: [],
}
