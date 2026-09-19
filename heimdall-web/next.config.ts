import type { NextConfig } from "next";
import path from "path";

const nextConfig: NextConfig = {
  // The Python pipeline lives in the parent folder; pin the workspace root to this app.
  turbopack: { root: path.join(__dirname) },
  // Standalone server bundle for Docker / offline deployment.
  output: "standalone",
};

export default nextConfig;
