"use client";

import { useCallback, useRef, useState } from "react";
import { cn } from "@/lib/utils";

export type UploadResult = {
  uri: string;
  filename: string;
  size: number;
  audio?: { sample_rate: number; channels: number; duration_s: number };
};

type FileItem = {
  file: File;
  status: "pending" | "uploading" | "done" | "error";
  progress: number;
  result?: UploadResult;
  error?: string;
};

export function AudioUploader({
  onUploaded,
  accept = "audio/*",
  prefix = "dataset_sources",
}: {
  onUploaded: (results: UploadResult[]) => void;
  accept?: string;
  prefix?: string;
}) {
  const [items, setItems] = useState<FileItem[]>([]);
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFiles = useCallback((fl: FileList | null) => {
    if (!fl) return;
    const next: FileItem[] = Array.from(fl).map((file) => ({
      file,
      status: "pending",
      progress: 0,
    }));
    setItems((prev) => [...prev, ...next]);
  }, []);

  async function uploadAll() {
    const results: UploadResult[] = [];
    for (let i = 0; i < items.length; i++) {
      const it = items[i];
      if (it.status === "done" && it.result) {
        results.push(it.result);
        continue;
      }
      setItems((cur) => cur.map((x, j) => (j === i ? { ...x, status: "uploading", progress: 0 } : x)));
      try {
        const form = new FormData();
        form.append("file", it.file);
        const r = await fetch(`/api/uploads?prefix=${encodeURIComponent(prefix)}`, {
          method: "POST",
          body: form,
        });
        if (!r.ok) throw new Error(`${r.status}: ${await r.text()}`);
        const body: UploadResult = await r.json();
        results.push(body);
        setItems((cur) =>
          cur.map((x, j) =>
            j === i ? { ...x, status: "done", progress: 100, result: body } : x,
          ),
        );
      } catch (e) {
        setItems((cur) =>
          cur.map((x, j) =>
            j === i ? { ...x, status: "error", error: (e as Error).message } : x,
          ),
        );
      }
    }
    onUploaded(results);
  }

  function clear() {
    setItems([]);
  }

  return (
    <div>
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragOver(false);
          handleFiles(e.dataTransfer.files);
        }}
        onClick={() => inputRef.current?.click()}
        className={cn(
          "rounded-lg border-2 border-dashed p-6 text-center cursor-pointer transition",
          dragOver ? "border-accent bg-accent/5" : "border-border hover:border-accent/60",
        )}
      >
        <div className="text-sm">Drop audio files here</div>
        <div className="text-xs text-muted mt-1">or click to browse</div>
        <input
          ref={inputRef}
          type="file"
          accept={accept}
          multiple
          className="hidden"
          onChange={(e) => handleFiles(e.target.files)}
        />
      </div>

      {items.length > 0 && (
        <div className="mt-3 space-y-1.5">
          {items.map((it, i) => (
            <div key={i} className="text-xs flex items-center justify-between border border-border rounded px-2 py-1.5">
              <span className="truncate flex-1 mr-2">{it.file.name}</span>
              <span className="text-muted">{(it.file.size / 1024 / 1024).toFixed(2)} MB</span>
              <span
                className={cn(
                  "ml-3 inline-flex items-center rounded-full px-2 py-0.5 text-[10px]",
                  it.status === "done" && "bg-emerald-500/15 text-emerald-400",
                  it.status === "uploading" && "bg-blue-500/15 text-blue-400",
                  it.status === "error" && "bg-red-500/15 text-red-400",
                  it.status === "pending" && "bg-zinc-500/15 text-zinc-400",
                )}
              >
                {it.status}
              </span>
            </div>
          ))}

          <div className="flex justify-end gap-2 pt-1">
            <button
              type="button"
              onClick={clear}
              className="px-2 py-1 text-xs rounded border border-border"
            >
              Clear
            </button>
            <button
              type="button"
              onClick={uploadAll}
              className="px-3 py-1 text-xs rounded bg-accent text-white font-medium"
            >
              Upload {items.filter((x) => x.status !== "done").length} file(s)
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
