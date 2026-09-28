import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Heimdall — DepthWizard",
  description: "Single-view satellite height estimation (DSM) and interactive 3D flythrough",
  icons: {
    icon: [
      { url: "/logo-32.png", sizes: "32x32", type: "image/png" },
      { url: "/logo-192.png", sizes: "192x192", type: "image/png" },
    ],
    apple: "/apple-icon.png",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body suppressHydrationWarning>{children}</body>
    </html>
  );
}
