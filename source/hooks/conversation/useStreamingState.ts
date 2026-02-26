import {useState, useEffect, useCallback} from 'react';
import type {UsageInfo} from '../../api/chat.js';

export type RetryStatus = {
	isRetrying: boolean;
	attempt: number;
	nextDelay: number;
	remainingSeconds?: number;
	errorMessage?: string;
};

export type CodebaseSearchStatus = {
	isSearching: boolean;
	attempt: number;
	maxAttempts: number;
	currentTopN: number;
	message: string;
	query?: string;
	originalResultsCount?: number;
	suggestion?: string;
};

export type StreamStatus = 'idle' | 'streaming' | 'stopping';

export type StreamingPhase =
	| 'thinking'
	| 'reasoning'
	| 'tooling'
	| 'waiting_retry'
	| 'finalizing';

export type StallLevel = 'none' | 'warning' | 'critical';

export type StallReason = 'no_token_no_event' | 'no_token' | 'no_event' | null;

const STALL_WARNING_SECONDS = 20;
const STALL_CRITICAL_SECONDS = 45;

export function useStreamingState() {
	const [streamStatus, setStreamStatus] = useState<StreamStatus>('idle');
	const isStreaming = streamStatus === 'streaming';
	const isStopping = streamStatus === 'stopping';

	const [streamTokenCount, setStreamTokenCountBase] = useState(0);
	const [isReasoning, setIsReasoningBase] = useState(false);
	const [abortController, setAbortController] =
		useState<AbortController | null>(null);
	const [contextUsage, setContextUsage] = useState<UsageInfo | null>(null);
	const [elapsedSeconds, setElapsedSeconds] = useState(0);
	const [timerStartTime, setTimerStartTime] = useState<number | null>(null);
	const [retryStatus, setRetryStatusBase] = useState<RetryStatus | null>(null);
	const [animationFrame, setAnimationFrame] = useState(0);
	const [codebaseSearchStatus, setCodebaseSearchStatus] =
		useState<CodebaseSearchStatus | null>(null);
	const [currentModel, setCurrentModel] = useState<string | null>(null);

	const [currentPhase, setCurrentPhase] = useState<StreamingPhase>('thinking');
	const [lastProgressAt, setLastProgressAt] = useState<number | null>(null);
	const [lastTokenAt, setLastTokenAt] = useState<number | null>(null);
	const [lastSubAgentEventAt, setLastSubAgentEventAt] = useState<number | null>(null);
	const [lastSubAgentEventType, setLastSubAgentEventType] =
		useState<string | null>(null);
	const [lastSubAgentName, setLastSubAgentName] = useState<string | null>(null);
	const [stallLevel, setStallLevel] = useState<StallLevel>('none');
	const [stallReason, setStallReason] = useState<StallReason>(null);

	const setIsStreaming: React.Dispatch<
		React.SetStateAction<boolean>
	> = action => {
		setStreamStatus(prev => {
			const currentIsStreaming = prev === 'streaming';
			const nextIsStreaming =
				typeof action === 'function' ? action(currentIsStreaming) : action;

			if (nextIsStreaming) return 'streaming';
			return 'idle';
		});
	};

	const setIsStopping: React.Dispatch<
		React.SetStateAction<boolean>
	> = action => {
		setStreamStatus(prev => {
			const currentIsStopping = prev === 'stopping';
			const nextIsStopping =
				typeof action === 'function' ? action(currentIsStopping) : action;

			if (nextIsStopping) return 'stopping';
			if (prev === 'stopping') return 'idle';
			return prev;
		});
	};

	const setCurrentPhaseSafe: React.Dispatch<
		React.SetStateAction<StreamingPhase>
	> = useCallback(action => {
		setCurrentPhase(action);
	}, []);

	const markSubAgentProgress = useCallback(
		(eventType: string, agentName?: string) => {
			const now = Date.now();
			setLastSubAgentEventAt(now);
			setLastProgressAt(now);
			setLastSubAgentEventType(eventType);
			if (agentName) setLastSubAgentName(agentName);

			if (
				eventType === 'tool_calls' ||
				eventType === 'tool_result' ||
				eventType === 'progress' ||
				eventType === 'context_usage' ||
				eventType === 'context_compressing' ||
				eventType === 'context_compressed' ||
				eventType === 'agent_spawned' ||
				eventType === 'spawned_agent_completed'
			) {
				setCurrentPhase('tooling');
			}
		},
		[],
	);

	const setStreamTokenCount: React.Dispatch<React.SetStateAction<number>> =
		action => {
			setStreamTokenCountBase(prev => {
				const next = typeof action === 'function' ? action(prev) : action;
				if (next > prev) {
					const now = Date.now();
					setLastTokenAt(now);
					setLastProgressAt(now);
				}
				return next;
			});
		};

	const setIsReasoning: React.Dispatch<React.SetStateAction<boolean>> =
		action => {
			setIsReasoningBase(action);
		};

	const setRetryStatus: React.Dispatch<React.SetStateAction<RetryStatus | null>> =
		action => {
			setRetryStatusBase(action);
		};

	const evaluateStall = useCallback(() => {
		if (!isStreaming) {
			setStallLevel('none');
			setStallReason(null);
			return;
		}

		const now = Date.now();
		const baseline = lastProgressAt ?? timerStartTime ?? now;
		const idleSeconds = Math.floor((now - baseline) / 1000);

		const hasToken = !!lastTokenAt;
		const hasSubAgentEvent = !!lastSubAgentEventAt;
		let nextReason: StallReason = null;
		if (!hasToken && !hasSubAgentEvent) nextReason = 'no_token_no_event';
		else if (!hasToken) nextReason = 'no_token';
		else if (!hasSubAgentEvent && currentPhase === 'tooling')
			nextReason = 'no_event';
		else if (!hasSubAgentEvent) nextReason = 'no_token';

		if (idleSeconds >= STALL_CRITICAL_SECONDS) {
			setStallLevel('critical');
			setStallReason(nextReason);
			return;
		}
		if (idleSeconds >= STALL_WARNING_SECONDS) {
			setStallLevel('warning');
			setStallReason(nextReason);
			return;
		}
		setStallLevel('none');
		setStallReason(null);
	}, [
		isStreaming,
		lastProgressAt,
		lastTokenAt,
		lastSubAgentEventAt,
		timerStartTime,
		currentPhase,
	]);

	useEffect(() => {
		if (!isStreaming) return;

		const interval = setInterval(() => {
			setAnimationFrame(prev => (prev + 1) % 2);
		}, 500);

		return () => {
			clearInterval(interval);
			setAnimationFrame(0);
		};
	}, [isStreaming]);

	useEffect(() => {
		if (isStreaming && timerStartTime === null) {
			const now = Date.now();
			setTimerStartTime(now);
			setElapsedSeconds(0);
			setCurrentPhase('thinking');
			setLastProgressAt(now);
			setLastTokenAt(null);
			setLastSubAgentEventAt(null);
			setLastSubAgentEventType(null);
			setLastSubAgentName(null);
			setStallLevel('none');
			setStallReason(null);
		} else if (!isStreaming && timerStartTime !== null) {
			setTimerStartTime(null);
			setCurrentPhase('thinking');
			setStallLevel('none');
			setStallReason(null);
		}
	}, [isStreaming, timerStartTime]);

	useEffect(() => {
		if (timerStartTime === null) return;

		const interval = setInterval(() => {
			const elapsed = Math.floor((Date.now() - timerStartTime) / 1000);
			setElapsedSeconds(elapsed);
		}, 1000);

		return () => clearInterval(interval);
	}, [timerStartTime]);

	useEffect(() => {
		if (!isStreaming) return;
		evaluateStall();
		const interval = setInterval(() => {
			evaluateStall();
		}, 1000);
		return () => clearInterval(interval);
	}, [isStreaming, evaluateStall]);

	useEffect(() => {
		if (!retryStatus?.isRetrying) return;
		if (retryStatus.remainingSeconds !== undefined) return;

		setRetryStatus(prev =>
			prev
				? {
						...prev,
						remainingSeconds: Math.ceil(prev.nextDelay / 1000),
				  }
				: null,
		);
	}, [retryStatus?.isRetrying]);

	useEffect(() => {
		if (!retryStatus || !retryStatus.isRetrying) return;
		if (retryStatus.remainingSeconds === undefined) return;

		const interval = setInterval(() => {
			setRetryStatus(prev => {
				if (!prev || prev.remainingSeconds === undefined) return prev;

				const newRemaining = prev.remainingSeconds - 1;
				if (newRemaining <= 0) {
					return {
						...prev,
						remainingSeconds: 0,
					};
				}

				return {
					...prev,
					remainingSeconds: newRemaining,
				};
			});
		}, 1000);

		return () => clearInterval(interval);
	}, [retryStatus?.isRetrying]);

	return {
		streamStatus,
		setStreamStatus,
		isStreaming,
		setIsStreaming,
		isStopping,
		setIsStopping,
		streamTokenCount,
		setStreamTokenCount,
		isReasoning,
		setIsReasoning,
		abortController,
		setAbortController,
		contextUsage,
		setContextUsage,
		elapsedSeconds,
		retryStatus,
		setRetryStatus,
		animationFrame,
		codebaseSearchStatus,
		setCodebaseSearchStatus,
		currentModel,
		setCurrentModel,
		currentPhase,
		setCurrentPhase: setCurrentPhaseSafe,
		lastProgressAt,
		lastTokenAt,
		lastSubAgentEventAt,
		lastSubAgentEventType,
		lastSubAgentName,
		stallLevel,
		stallReason,
		markSubAgentProgress,
		evaluateStall,
	};
}
