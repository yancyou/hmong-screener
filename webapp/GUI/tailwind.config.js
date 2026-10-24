/** @type {import('tailwindcss').Config} */
import defaultTheme from 'tailwindcss/defaultTheme'

export default {
  darkMode: 'class',
  content: [
    "./index.html",
    "./main.jsx",
    "./hmong_audio_similarity_ui_react.jsx",
    "./DataCollectionPage.jsx",
    "./BatchScorePage.jsx",
    "./ui.jsx",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', ...defaultTheme.fontFamily.sans],
      },
      colors: {
        brand: {
          DEFAULT: '#C5050C',
          light: '#Ea4D53',
        }
      }
    },
  },
  plugins: [],
}
