import React from 'react';
import {Box, Text} from 'ink';
import {useTheme} from '../../contexts/ThemeContext.js';
import {useI18n} from '../../../i18n/I18nContext.js';
import ShimmerText from '../common/ShimmerText.js';
import CodebaseSearchStatus from './CodebaseSearchStatus.js';
import {formatElapsedTime} from '../../../utils/core/textUtils.js';

/**
 * 截断错误消息，避免过长显示
 */
function truncateErrorMessage(
	message: string,
	maxLength: number = 100,
): string {
	if (message.length <= maxLength) {
		return message;
	}
	return message.substring(0, maxLength) + '...';
}

type LoadingIndicatorProps = {
	isStreaming: boolean;
	isStopping: boolean;
	isSaving: boolean;
	hasPendingToolConfirmation: boolean;
	hasPendingUserQuestion: boolean;
	hasBlockingOverlay: boolean;
	terminalWidth: number;
	animationFrame: number;
	retryStatus: {
		isRetrying: boolean;
		errorMessage?: string;
		remainingSeconds?: number;
		attempt: number;
	} | null;
	codebaseSearchStatus: {
		isSearching: boolean;
		attempt: number;
		maxAttempts: number;
		currentTopN: number;
		message: string;
		query?: string;
		originalResultsCount?: number;
		suggestion?: string;
	} | null;
	isReasoning: boolean;
	streamTokenCount: number;
	elapsedSeconds: number;
	currentModel?: string | null;
	currentPhase:
		| 'thinking'
		| 'reasoning'
		| 'tooling'
		| 'waiting_retry'
		| 'finalizing';
	stallLevel: 'none' | 'warning' | 'critical';
	stallReason: 'no_token_no_event' | 'no_token' | 'no_event' | null;
	lastProgressAt: number | null;
	lastSubAgentEventType: string | null;
	lastSubAgentName: string | null;
};

export default function LoadingIndicator({
	isStreaming,
	isStopping,
	isSaving,
	hasPendingToolConfirmation,
	hasPendingUserQuestion,
	hasBlockingOverlay,
	terminalWidth,
	animationFrame,
	retryStatus,
	codebaseSearchStatus,
	isReasoning,
	streamTokenCount,
	elapsedSeconds,
	currentModel,
	currentPhase,
	stallLevel,
	stallReason,
	lastProgressAt,
	lastSubAgentEventType,
	lastSubAgentName,
}: LoadingIndicatorProps) {
	const {theme} = useTheme();
	const {t} = useI18n();

	const getPhaseText = (): string => {
		if (currentPhase === 'reasoning' || isReasoning) {
			return t.chatScreen.statusDeepThinking;
		}
		if (currentPhase === 'tooling') {
			return t.chatScreen.statusExecutingTools;
		}
		if (currentPhase === 'waiting_retry') {
			return t.chatScreen.statusWaitingRetry;
		}
		if (currentPhase === 'finalizing') {
			return t.chatScreen.statusFinalizing;
		}
		return t.chatScreen.statusThinking;
	};

	const getStallReasonText = (): string => {
		if (stallReason === 'no_token') return t.chatScreen.statusNoNewTokens;
		if (stallReason === 'no_event')
			return t.chatScreen.statusNoNewSubAgentEvents;
		if (stallReason === 'no_token_no_event')
			return t.chatScreen.statusNoNewTokensOrEvents;
		return t.chatScreen.statusNoRecentProgress;
	};

	const stalledForSeconds = lastProgressAt
		? Math.max(0, Math.floor((Date.now() - lastProgressAt) / 1000))
		: 0;

	// 不显示加载指示器的条件
	if (
		(!isStreaming && !isSaving && !isStopping) ||
		hasPendingToolConfirmation ||
		hasPendingUserQuestion ||
		hasBlockingOverlay
	) {
		return null;
	}

	return (
		<Box marginBottom={1} paddingX={1} width={terminalWidth}>
			<Text color={['#00FFFF', '#1ACEB0'][animationFrame % 2] as any} bold>
				❆
			</Text>
			<Box marginLeft={1} flexDirection="column">
				{isStopping ? (
					<Text color={theme.colors.menuSecondary} dimColor>
						{t.chatScreen.statusStopping}
					</Text>
				) : isStreaming ? (
					<>
						{retryStatus && retryStatus.isRetrying ? (
							<Box flexDirection="column">
								{retryStatus.errorMessage && (
									<Text color="red" dimColor>
										{t.chatScreen.retryError.replace(
											'{message}',
											truncateErrorMessage(retryStatus.errorMessage),
										)}
									</Text>
								)}
								{retryStatus.remainingSeconds !== undefined &&
								retryStatus.remainingSeconds > 0 ? (
									<Text color="yellow" dimColor>
										{t.chatScreen.retryAttempt
											.replace('{current}', String(retryStatus.attempt))
											.replace('{max}', '5')}{' '}
										{t.chatScreen.retryIn.replace(
											'{seconds}',
											String(retryStatus.remainingSeconds),
										)}
									</Text>
								) : (
									<Text color="yellow" dimColor>
										{t.chatScreen.retryResending
											.replace('{current}', String(retryStatus.attempt))
											.replace('{max}', '5')}
									</Text>
								)}
							</Box>
						) : codebaseSearchStatus?.isSearching ? (
							<CodebaseSearchStatus status={codebaseSearchStatus} />
						) : (
							<Box flexDirection="column">
								<Text color={theme.colors.menuSecondary} dimColor bold>
									<ShimmerText text={getPhaseText()} />
									(
									{currentModel && (
										<>
											{currentModel}
											{' · '}
										</>
									)}
									{formatElapsedTime(elapsedSeconds)}
									{' · '}
									<Text color="cyan">
										↓{' '}
										{streamTokenCount >= 1000
											? `${(streamTokenCount / 1000).toFixed(1)}k`
											: streamTokenCount}{' '}
										tokens
									</Text>
									{lastSubAgentName && (
										<>
											{' · '}
											<Text color="magenta">
												{subAgentLabel(
													lastSubAgentName,
													lastSubAgentEventType,
													t.chatScreen.statusSubAgentLabel,
													t.chatScreen.statusSubAgentLabelWithEvent,
												)}
											</Text>
										</>
									)}
									)
								</Text>
								{stallLevel !== 'none' && (
									<Text
										color={stallLevel === 'critical' ? 'red' : 'yellow'}
										dimColor={stallLevel !== 'critical'}
									>
										{stallLevel === 'critical'
											? t.chatScreen.statusPotentiallyStalled
											: t.chatScreen.statusSlowProgress}{' '}
										· {getStallReasonText()} · {stalledForSeconds}s ·
										{t.chatScreen.statusPressEscToInterrupt}
									</Text>
								)}
							</Box>
						)}
					</>
				) : (
					<Text color={theme.colors.menuSecondary} dimColor>
						{t.chatScreen.sessionCreating}
					</Text>
				)}
			</Box>
		</Box>
	);
}

function subAgentLabel(
	agentName: string,
	eventType: string | null,
	labelTemplate: string,
	labelWithEventTemplate: string,
): string {
	if (!eventType) {
		return labelTemplate.replace('{agentName}', agentName);
	}
	return labelWithEventTemplate
		.replace('{agentName}', agentName)
		.replace('{eventType}', eventType);
}
