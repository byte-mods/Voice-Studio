"use client";

import React, { useState, useMemo } from "react";
import { SENTENCE_DICTS, PromptSentence } from "@/lib/sentences";

interface PromptDictionaryProps {
  onSelectSentence: (text: string, emotion: string) => void;
}

export function PromptDictionary({ onSelectSentence }: PromptDictionaryProps) {
  const [activeLang, setActiveLang] = useState("en");
  const [searchQuery, setSearchQuery] = useState("");

  const selectedLangSheet = useMemo(() => {
    return SENTENCE_DICTS.find((d) => d.value === activeLang) || SENTENCE_DICTS[0];
  }, [activeLang]);

  const [activeCategory, setActiveCategory] = useState("");

  // Auto-reset active category when language changes or initialize it
  const categories = selectedLangSheet.categories;
  const currentCategory = useMemo(() => {
    if (!activeCategory || !categories.some((c) => c.name === activeCategory)) {
      return categories[0]?.name ?? "";
    }
    return activeCategory;
  }, [activeCategory, categories]);

  const filteredSentences = useMemo(() => {
    const group = categories.find((c) => c.name === currentCategory);
    if (!group) return [];
    if (!searchQuery.trim()) return group.sentences;

    const query = searchQuery.toLowerCase();
    return group.sentences.filter(
      (s) =>
        s.text.toLowerCase().includes(query) ||
        (s.translation && s.translation.toLowerCase().includes(query))
    );
  }, [categories, currentCategory, searchQuery]);

  return (
    <div className="flex flex-col h-full space-y-3">
      {/* Language Selector Tabs */}
      <div className="flex bg-card/60 border border-border p-1 rounded-lg gap-1">
        {SENTENCE_DICTS.map((lang) => {
          const active = lang.value === activeLang;
          return (
            <button
              key={lang.value}
              type="button"
              onClick={() => {
                setActiveLang(lang.value);
                setSearchQuery("");
                setActiveCategory(""); // triggers reset
              }}
              className={`flex-1 flex items-center justify-center gap-1.5 py-1.5 rounded-md text-xs font-semibold transition ${
                active
                  ? "bg-accent text-white shadow-md shadow-accent/20"
                  : "text-muted hover:text-white hover:bg-card"
              }`}
            >
              <span>{lang.flag}</span>
              <span>{lang.name}</span>
            </button>
          );
        })}
      </div>

      {/* Category Dropdown and Search Input */}
      <div className="space-y-2">
        <div>
          <label className="block text-[10px] uppercase tracking-wider text-muted mb-1">Category</label>
          <select
            className="w-full bg-card border border-border rounded-md px-2.5 py-1.5 text-xs text-white outline-none focus:border-accent"
            value={currentCategory}
            onChange={(e) => setActiveCategory(e.target.value)}
          >
            {categories.map((c) => (
              <option key={c.name} value={c.name}>
                {c.name}
              </option>
            ))}
          </select>
        </div>

        <div>
          <input
            type="text"
            className="w-full bg-card border border-border rounded-md px-2.5 py-1.5 text-xs text-white placeholder:text-muted outline-none focus:border-accent transition"
            placeholder="Search prompts..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
        </div>
      </div>

      {/* Sentences List */}
      <div className="flex-1 overflow-y-auto max-h-[350px] space-y-2 pr-1 custom-scrollbar">
        {filteredSentences.length === 0 ? (
          <div className="text-center py-6 text-xs text-muted">No sentences match your search</div>
        ) : (
          filteredSentences.map((s, idx) => (
            <button
              key={idx}
              type="button"
              onClick={() => onSelectSentence(s.text, s.emotion)}
              className="w-full text-left bg-card/40 hover:bg-card/90 border border-border hover:border-accent/40 rounded-lg p-2.5 transition flex flex-col space-y-1.5 group relative overflow-hidden"
            >
              {/* Text */}
              <p className="text-sm font-medium text-white/90 group-hover:text-white transition leading-relaxed">
                {s.text}
              </p>

              {/* Translation (for Hindi/Hinglish) */}
              {s.translation && (
                <p className="text-xs text-muted leading-relaxed font-normal italic">
                  {s.translation}
                </p>
              )}

              {/* Bottom Metadata Info */}
              <div className="flex items-center justify-between pt-1">
                <span
                  className={`text-[9px] uppercase font-bold tracking-wider px-2 py-0.5 rounded-full border ${
                    {
                      neutral: "text-muted bg-card border-border",
                      calm: "text-sky-400 bg-sky-500/10 border-sky-500/20",
                      cheerful: "text-amber-400 bg-amber-500/10 border-amber-500/20",
                      happy: "text-emerald-400 bg-emerald-500/10 border-emerald-500/20",
                      excited: "text-fuchsia-400 bg-fuchsia-500/10 border-fuchsia-500/20",
                      concerned: "text-red-400 bg-red-500/10 border-red-500/20",
                    }[s.emotion] || "text-muted bg-card border-border"
                  }`}
                >
                  🎭 {s.emotion}
                </span>
                <span className="text-[10px] text-accent font-medium opacity-0 group-hover:opacity-100 transition">
                  Use Sentence →
                </span>
              </div>
            </button>
          ))
        )}
      </div>

      <style jsx global>{`
        .custom-scrollbar::-webkit-scrollbar {
          width: 4px;
        }
        .custom-scrollbar::-webkit-scrollbar-track {
          background: rgba(var(--bg), 0.1);
        }
        .custom-scrollbar::-webkit-scrollbar-thumb {
          background: rgba(var(--border), 0.5);
          border-radius: 2px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover {
          background: rgb(var(--accent));
        }
      `}</style>
    </div>
  );
}
