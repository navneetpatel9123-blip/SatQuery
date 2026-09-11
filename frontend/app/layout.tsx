import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import "./globals.css";

const inter = Inter({
  variable: "--font-geist-sans",
  subsets: ["latin"],
  display: "swap",
});

const jetbrainsMono = JetBrains_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
  display: "swap",
});

export const metadata: Metadata = {
  title: "SatQuery AI — Bi-Temporal Satellite Change Detection & VQA",
  description:
    "Advanced geospatial AI platform for bi-temporal satellite image change detection, understanding, and visual question answering with full execution-trace explainability.",
  keywords: [
    "satellite imagery",
    "change detection",
    "GeoTIFF",
    "VQA",
    "geospatial AI",
    "remote sensing",
  ],
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html
      lang="en"
      className={`${inter.variable} ${jetbrainsMono.variable}`}
    >
      <body>
        {/* Background effects */}
        <div className="bg-grid" aria-hidden="true" />
        <div className="bg-orb bg-orb--cyan" aria-hidden="true" />
        <div className="bg-orb bg-orb--purple" aria-hidden="true" />
        <div className="bg-orb bg-orb--emerald" aria-hidden="true" />
        {children}
      </body>
    </html>
  );
}
