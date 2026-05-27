"use client";

import useSWR from "swr";
import { me, type AuthUser } from "@/lib/auth";

export function useCurrentUser(): {
  user: AuthUser | null;
  isLoading: boolean;
  reload: () => void;
} {
  const { data, isLoading, mutate } = useSWR<AuthUser | null>("me", () => me(), {
    revalidateOnFocus: false,
    shouldRetryOnError: false,
  });
  return { user: data ?? null, isLoading, reload: () => mutate() };
}
