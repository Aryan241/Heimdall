import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Heimdall — DepthWizard",
  description: "Single-view satellite height estimation (DSM) and interactive 3D flythrough",
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
