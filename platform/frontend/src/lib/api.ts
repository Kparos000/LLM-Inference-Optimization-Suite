import type { PlatformResponse } from "./types";

const API_BASE = process.env.NEXT_PUBLIC_PLATFORM_API_BASE ?? "http://127.0.0.1:8000";

export async function fetchPlatform<T>(path: string): Promise<T | null> {
  try {
    const response = await fetch(`${API_BASE}${path}`, { cache: "no-store" });
    if (!response.ok) {
      return null;
    }
    const payload = (await response.json()) as PlatformResponse<T>;
    return payload.data;
  } catch {
    return null;
  }
}

