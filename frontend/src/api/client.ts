import type { BodyProfile } from "../types";

const CLIENT_REF_KEY = "sizesync_client_ref";

function getClientRef(): string {
  let ref = localStorage.getItem(CLIENT_REF_KEY);
  if (!ref) {
    ref = crypto.randomUUID();
    localStorage.setItem(CLIENT_REF_KEY, ref);
  }
  return ref;
}

export class ApiError extends Error {
  constructor(message: string, public status: number) {
    super(message);
  }
}

async function handle<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new ApiError(body.detail ?? `Request failed (${res.status})`, res.status);
  }
  return res.json();
}

export async function createProfile(params: {
  heightCm: number;
  front: File;
  side: File;
  back: File;
}): Promise<BodyProfile> {
  const form = new FormData();
  form.append("client_ref", getClientRef());
  form.append("height_cm", String(params.heightCm));
  form.append("front", params.front);
  form.append("side", params.side);
  form.append("back", params.back);

  const res = await fetch("/api/profiles", { method: "POST", body: form });
  return handle<BodyProfile>(res);
}

export async function getProfile(id: string): Promise<BodyProfile> {
  const res = await fetch(`/api/profiles/${id}`);
  return handle<BodyProfile>(res);
}

export async function startReconstruction(id: string): Promise<BodyProfile> {
  const res = await fetch(`/api/profiles/${id}/reconstruct`, { method: "POST" });
  return handle<BodyProfile>(res);
}
