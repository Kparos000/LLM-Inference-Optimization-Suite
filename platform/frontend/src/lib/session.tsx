"use client";

import { createContext, useContext, useEffect, useMemo, useState } from "react";
import type { ChapterId, ExperimentSession, ResultType } from "./types";

const STORAGE_KEY = "inference-platform-session-v1";

const defaultSession: ExperimentSession = {
  baselineRunId: "main_inference_v1",
  currentChapter: "about",
  selectedMandatoryRepairs: [],
  selectedCoreOptimizations: [],
  validatedRecipe: null,
  resultType: "measured",
  selectedScenarioId: "main_inference_v1",
  selectedOptimizedRunId: null
};

type SessionContextValue = {
  session: ExperimentSession;
  setChapter: (chapter: ChapterId, resultType?: ResultType) => void;
  toggleMandatoryRepair: (id: string) => void;
  toggleCoreOptimization: (id: string) => void;
  applyAllSelected: (mandatoryIds: string[]) => void;
};

const SessionContext = createContext<SessionContextValue | null>(null);

function toggle(list: string[], value: string): string[] {
  return list.includes(value) ? list.filter((item) => item !== value) : [...list, value];
}

export function ExperimentSessionProvider({ children }: { children: React.ReactNode }) {
  const [session, setSession] = useState<ExperimentSession>(defaultSession);

  useEffect(() => {
    const saved = window.localStorage.getItem(STORAGE_KEY);
    if (saved) {
      setSession({ ...defaultSession, ...(JSON.parse(saved) as Partial<ExperimentSession>) });
    }
  }, []);

  useEffect(() => {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(session));
  }, [session]);

  const value = useMemo<SessionContextValue>(
    () => ({
      session,
      setChapter: (chapter, resultType = "measured") => {
        setSession((current) => ({ ...current, currentChapter: chapter, resultType }));
      },
      toggleMandatoryRepair: (id) => {
        setSession((current) => ({
          ...current,
          selectedMandatoryRepairs: toggle(current.selectedMandatoryRepairs, id),
          resultType: "planned"
        }));
      },
      toggleCoreOptimization: (id) => {
        setSession((current) => ({
          ...current,
          selectedCoreOptimizations: toggle(current.selectedCoreOptimizations, id),
          resultType: "planned"
        }));
      },
      applyAllSelected: (mandatoryIds) => {
        setSession((current) => ({
          ...current,
          selectedMandatoryRepairs: Array.from(new Set(mandatoryIds)),
          validatedRecipe: "plan_only_apply_all_selected",
          resultType: "planned",
          selectedScenarioId: "planned_optimized_recipe"
        }));
      }
    }),
    [session]
  );

  return <SessionContext.Provider value={value}>{children}</SessionContext.Provider>;
}

export function useExperimentSession() {
  const context = useContext(SessionContext);
  if (!context) {
    throw new Error("useExperimentSession must be used inside ExperimentSessionProvider");
  }
  return context;
}

