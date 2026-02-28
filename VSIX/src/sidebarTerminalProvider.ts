import * as vscode from 'vscode';
import {PtyManager, ShellType} from './ptyManager';

type LaunchPolicy = 'ensure' | 'restart';
type Trigger =
	| 'viewReady'
	| 'viewRecreate'
	| 'openOrFocus'
	| 'manualRestart'
	| 'visibility'
	| 'configChange';

type LifecycleAction = {
	policy: LaunchPolicy;
	focus: boolean;
	requestWebviewFocus: boolean;
	resetFrontend: boolean;
	suppressExitBanner: boolean;
};
type EnsureOptions = {focus?: boolean};
type RestartOptions = {reason?: 'manualRestart' | 'configChange'};

type TerminalConfig = {
	shellType: ShellType;
	fontFamily: string;
	fontSize: number;
	fontWeight: string;
	lineHeight: number;
};

type NormalizedFontConfig = Omit<TerminalConfig, 'shellType'>;

type ExtensionToWebviewMessage =
	| {type: 'output'; data: string}
	| {type: 'clear'}
	| {type: 'reset'}
	| {type: 'fit'}
	| {type: 'focus'}
	| {
			type: 'updateFont';
			fontFamily: string;
			fontSize: number;
			fontWeight: string;
			lineHeight: number;
	  }
	| {type: 'exit'; code: number}
	| {type: 'fileDrop'; paths: string[]};

type WebviewToExtensionMessage =
	| {type: 'ready'}
	| {type: 'input'; data: string}
	| {type: 'resize'; cols: number; rows: number}
	| {type: 'rendererStall'; reason?: string};

const RESOURCE_ROOT_SEGMENTS: readonly (readonly string[])[] = [
	['res'],
	['node_modules', '@xterm'],
];

const XTERM_SCRIPT_SEGMENTS: readonly (readonly string[])[] = [
	['node_modules', '@xterm', 'xterm', 'lib', 'xterm.js'],
	['node_modules', '@xterm', 'addon-fit', 'lib', 'addon-fit.js'],
	['node_modules', '@xterm', 'addon-web-links', 'lib', 'addon-web-links.js'],
	['node_modules', '@xterm', 'addon-search', 'lib', 'addon-search.js'],
	['node_modules', '@xterm', 'addon-webgl', 'lib', 'addon-webgl.js'],
	['node_modules', '@xterm', 'addon-unicode11', 'lib', 'addon-unicode11.js'],
];

const XTERM_CSS_SEGMENTS = [
	'node_modules',
	'@xterm',
	'xterm',
	'css',
	'xterm.css',
] as const;
const SIDEBAR_STYLE_SEGMENTS = ['res', 'sidebarTerminal.css'] as const;
const SIDEBAR_SCRIPT_SEGMENTS = ['res', 'sidebarTerminal.js'] as const;

const OUTPUT_FLUSH_INTERVAL_MS = 16;
const OUTPUT_IMMEDIATE_FLUSH_THRESHOLD = 16 * 1024;
const OUTPUT_BUFFER_MAX_BYTES = 2 * 1024 * 1024;
const OUTPUT_TRUNCATION_NOTICE =
	'\r\n[Output truncated while terminal view was unavailable]\r\n';
const FOCUS_RETRY_DELAYS_MS = [0, 80, 240] as const;

const FONT_SIZE_MIN = 8;
const FONT_SIZE_MAX = 32;
const LINE_HEIGHT_MIN = 0.8;
const LINE_HEIGHT_MAX = 2.0;


const DEFAULT_ACTION: LifecycleAction = {
	policy: 'ensure',
	focus: false,
	requestWebviewFocus: false,
	resetFrontend: false,
	suppressExitBanner: false,
};

const TRIGGER_ACTIONS: Record<Trigger, LifecycleAction> = {
	viewReady: {
		...DEFAULT_ACTION,
	},
	visibility: {
		...DEFAULT_ACTION,
		requestWebviewFocus: true,
	},
	openOrFocus: {
		...DEFAULT_ACTION,
		focus: true,
		requestWebviewFocus: true,
	},
	manualRestart: {
		policy: 'restart',
		focus: false,
		requestWebviewFocus: true,
		resetFrontend: true,
		suppressExitBanner: true,
		},
	viewRecreate: {
		policy: 'restart',
		focus: false,
		requestWebviewFocus: false,
		resetFrontend: true,
		suppressExitBanner: true,
	},
	configChange: {
		policy: 'restart',
		focus: false,
		requestWebviewFocus: false,
		resetFrontend: true,
		suppressExitBanner: false,
	},
};

function quotePathIfNeeded(path: string): string {
	return path.includes(' ') ? `"${path}"` : path;
}

function formatPathPayload(paths: readonly string[]): string {
	return paths.map(quotePathIfNeeded).join(' ');
}

function isRecord(value: unknown): value is Record<string, unknown> {
	return typeof value === 'object' && value !== null;
}

function clampNumber(value: number, min: number, max: number): number {
	return Math.max(min, Math.min(max, value));
}

function parseWebviewMessage(rawMessage: unknown): WebviewToExtensionMessage | undefined {
	if (!isRecord(rawMessage) || typeof rawMessage.type !== 'string') {
		return undefined;
	}

	switch (rawMessage.type) {
		case 'ready':
			return {type: 'ready'};
		case 'input':
			if (typeof rawMessage.data === 'string') {
				return {type: 'input', data: rawMessage.data};
			}
			return undefined;
		case 'resize':
			if (
				typeof rawMessage.cols === 'number' &&
				typeof rawMessage.rows === 'number' &&
				Number.isFinite(rawMessage.cols) &&
				Number.isFinite(rawMessage.rows)
			) {
				return {
					type: 'resize',
					cols: rawMessage.cols,
					rows: rawMessage.rows,
				};
			}
			return undefined;
		case 'rendererStall':
			if (
				typeof rawMessage.reason === 'undefined' ||
				typeof rawMessage.reason === 'string'
			) {
				return {type: 'rendererStall', reason: rawMessage.reason};
			}
			return undefined;
		default:
			return undefined;
	}
}

function mergeActions(
	base: LifecycleAction,
	incoming: LifecycleAction,
): LifecycleAction {
	const policy: LaunchPolicy =
		base.policy === 'restart' || incoming.policy === 'restart'
			? 'restart'
			: 'ensure';

	return {
		policy,
		focus: base.focus || incoming.focus,
		requestWebviewFocus:
			base.requestWebviewFocus || incoming.requestWebviewFocus,
		resetFrontend: base.resetFrontend || incoming.resetFrontend,
		suppressExitBanner: base.suppressExitBanner || incoming.suppressExitBanner,
	};
}

class PendingLifecycleQueue {
	private pendingAction: LifecycleAction | undefined;

	public queue(action: LifecycleAction): void {
		this.pendingAction = this.pendingAction
			? mergeActions(this.pendingAction, action)
			: {...action};
	}

	public mergeWithPending(current: LifecycleAction): LifecycleAction {
		if (!this.pendingAction) {
			return current;
		}
		const merged = mergeActions(this.pendingAction, current);
		this.pendingAction = undefined;
		return merged;
	}

	public clear(): void {
		this.pendingAction = undefined;
	}
}

export class SidebarTerminalProvider implements vscode.WebviewViewProvider {
	public static readonly viewType = 'snowCliTerminal';

	private view?: vscode.WebviewView;
	private readonly ptyManager: PtyManager;
	private readonly lifecycleQueue = new PendingLifecycleQueue();
	private startupCommand: string;
	private webviewReady = false;
	private hasResolvedViewOnce = false;
	private ensureRunningTimer: NodeJS.Timeout | undefined;
	private latestTerminalSize: {cols: number; rows: number} | undefined;
	private terminalSessionNonce = 0;
	private suppressedExitSessionNonces = new Set<number>();
	private outputChunks: string[] = [];
	private outputBytes = 0;
	private outputTruncated = false;
	private outputFlushTimer: NodeJS.Timeout | undefined;
	private focusRetryTimers = new Set<NodeJS.Timeout>();
	private lastRendererStallNoticeAt = 0;

	constructor(
		private readonly extensionUri: vscode.Uri,
		startupCommand?: string,
	) {
		this.ptyManager = new PtyManager();
		this.startupCommand = startupCommand ?? 'snow';
		this.applyShellType();
	}

	private getTerminalConfig(): TerminalConfig {
		const cfg = vscode.workspace.getConfiguration('snow-cli.terminal');
		return {
			shellType: cfg.get<ShellType>('shellType', 'auto'),
			fontFamily: cfg.get<string>('fontFamily', ''),
			fontSize: cfg.get<number>('fontSize', 14),
			fontWeight: cfg.get<string>('fontWeight', 'normal'),
			lineHeight: cfg.get<number>('lineHeight', 1.0),
		};
	}

	private normalizeFontConfig(config: TerminalConfig): NormalizedFontConfig {
		return {
			fontFamily: config.fontFamily || 'monospace',
			fontSize: clampNumber(config.fontSize, FONT_SIZE_MIN, FONT_SIZE_MAX),
			fontWeight: config.fontWeight || 'normal',
			lineHeight: clampNumber(
				config.lineHeight,
				LINE_HEIGHT_MIN,
				LINE_HEIGHT_MAX,
			),
		};
	}

	private applyShellType(): void {
		const {shellType} = this.getTerminalConfig();
		this.ptyManager.setShellType(shellType);
	}

	private sendFontConfig(): void {
		const normalized = this.normalizeFontConfig(this.getTerminalConfig());
		this.postWebviewMessage({type: 'updateFont', ...normalized});
	}

	private getWorkspaceFolderForActiveEditor(): string | undefined {
		const editor = vscode.window.activeTextEditor;
		const folder = editor
			? vscode.workspace.getWorkspaceFolder(editor.document.uri)
			: undefined;
		return (
			folder?.uri.fsPath ?? vscode.workspace.workspaceFolders?.[0]?.uri.fsPath
		);
	}

	public setStartupCommand(command: string): void {
		this.startupCommand = command;
	}

	public ensureTerminal(options?: EnsureOptions): void {
		this.runLifecycleAction('openOrFocus', options);
	}

	public restartTerminal(options?: RestartOptions): void {
		this.runLifecycleAction(options?.reason ?? 'manualRestart');
	}

	public onViewReady(): void {
		this.webviewReady = true;
		this.runLifecycleAction('viewReady');
		this.sendFontConfig();
		this.flushOutputBuffer();
	}

	public onViewRecreate(): void {
		this.runLifecycleAction('viewRecreate');
	}

	public resolveWebviewView(
		webviewView: vscode.WebviewView,
		_context: vscode.WebviewViewResolveContext,
		_token: vscode.CancellationToken,
	): void {
		const isViewRecreate = this.hasResolvedViewOnce;
		this.hasResolvedViewOnce = true;
		this.view = webviewView;
		this.webviewReady = false;

		this.configureWebview(webviewView);
		this.registerWebviewEventHandlers(webviewView);
		if (isViewRecreate) {
			this.onViewRecreate();
		}
	}

	private configureWebview(webviewView: vscode.WebviewView): void {
		webviewView.webview.options = {
			enableScripts: true,
			localResourceRoots: RESOURCE_ROOT_SEGMENTS.map(segments =>
				this.getExtensionResourceUri(segments),
			),
		};
		webviewView.webview.html = this.getHtmlForWebview(webviewView.webview);
	}

	private registerWebviewEventHandlers(webviewView: vscode.WebviewView): void {
		webviewView.webview.onDidReceiveMessage(message => {
			this.handleMessage(message);
		});

		webviewView.onDidChangeVisibility(() => {
			if (webviewView.visible) {
				this.scheduleEnsureRunning();
			}
		});

		webviewView.onDidDispose(() => {
			this.handleViewDisposed();
		});
	}

	private teardownRuntimeState(): void {
		this.clearEnsureRunningTimer();
		this.clearFocusRetryTimers();
		this.clearOutputBuffer();
		this.suppressedExitSessionNonces.clear();
		this.lifecycleQueue.clear();
		this.ptyManager.kill();
	}

	private handleViewDisposed(): void {
		this.webviewReady = false;
		this.view = undefined;
		this.teardownRuntimeState();
	}

	private isWebviewOperational(): boolean {
		return Boolean(this.view && this.webviewReady);
	}

	private handleMessage(rawMessage: unknown): void {
		const message = parseWebviewMessage(rawMessage);
		if (!message) {
			return;
		}

		switch (message.type) {
			case 'ready':
				this.onViewReady();
				return;
			case 'input':
				this.handleInputMessage(message.data);
				return;
			case 'resize':
				this.handleResizeMessage(message.cols, message.rows);
				return;
			case 'rendererStall':
				this.handleRendererStallMessage(message.reason);
				return;
		}
	}

	private handleInputMessage(data: string): void {
		if (!data) {
			return;
		}
		this.writeInputToTerminal(data);
	}

	private handleResizeMessage(cols: number, rows: number): void {
		const nextCols = Math.floor(cols);
		const nextRows = Math.floor(rows);
		if (nextCols <= 0 || nextRows <= 0) {
			return;
		}
		this.latestTerminalSize = {cols: nextCols, rows: nextRows};
		this.ptyManager.resize(nextCols, nextRows);
	}

	private handleRendererStallMessage(reason?: string): void {
		this.postWebviewMessage({type: 'fit'});
		this.requestWebviewFocus();
		const now = Date.now();
		if (now - this.lastRendererStallNoticeAt >= 10000) {
			this.lastRendererStallNoticeAt = now;
			void vscode.window.setStatusBarMessage(
				`Snow CLI: terminal renderer recovered${reason ? ` (${reason})` : ''}.`,
				3000,
			);
		}
	}

	private writeInputToTerminal(data: string): void {
		this.ensureTerminalRunning();
		this.ptyManager.write(data);
	}

	private startTerminal(): void {
		this.applyShellType();
		const workspaceFolder = this.getWorkspaceFolderForActiveEditor();
		const cwd = workspaceFolder || process.cwd();
		const sessionNonce = ++this.terminalSessionNonce;

		this.ptyManager.start(
			cwd,
			{
				onData: data => {
					this.enqueueOutput(data);
				},
				onExit: code => {
					this.handleTerminalExit(sessionNonce, code);
				},
			},
			this.startupCommand,
			this.latestTerminalSize,
		);
	}

	private handleTerminalExit(
		sessionNonce: number,
		code: number,
	): void {
		if (this.suppressedExitSessionNonces.delete(sessionNonce)) {
			return;
		}

		this.flushOutputBuffer();
		this.postWebviewMessage({type: 'exit', code});
	}

	private enqueueOutput(data: string): void {
		if (!data) {
			return;
		}

		this.outputChunks.push(data);
		this.outputBytes += data.length;
		this.enforceOutputBufferLimit();

		if (this.outputBytes >= OUTPUT_IMMEDIATE_FLUSH_THRESHOLD) {
			this.flushOutputBuffer();
			return;
		}
		if (this.outputFlushTimer) {
			return;
		}

		this.outputFlushTimer = setTimeout(() => {
			this.outputFlushTimer = undefined;
			this.flushOutputBuffer();
		}, OUTPUT_FLUSH_INTERVAL_MS);
	}

	private flushOutputBuffer(): void {
		this.clearOutputFlushTimer();
		if (this.outputChunks.length === 0) {
			return;
		}
		if (!this.isWebviewOperational()) {
			return;
		}

		const data = this.outputChunks.join('');
		const payload = this.outputTruncated
			? `${OUTPUT_TRUNCATION_NOTICE}${data}`
			: data;
		this.resetOutputBufferState();
		this.postWebviewMessage({type: 'output', data: payload});
	}

	private clearOutputBuffer(): void {
		this.clearOutputFlushTimer();
		this.resetOutputBufferState();
	}

	private resetOutputBufferState(): void {
		this.outputChunks = [];
		this.outputBytes = 0;
		this.outputTruncated = false;
	}

	private enforceOutputBufferLimit(): void {
		if (this.outputBytes <= OUTPUT_BUFFER_MAX_BYTES) {
			return;
		}
		const fullData = this.outputChunks.join('');
		const tail = fullData.slice(-OUTPUT_BUFFER_MAX_BYTES);
		this.outputChunks = [tail];
		this.outputBytes = tail.length;
		this.outputTruncated = true;
	}

	private scheduleEnsureRunning(): void {
		if (!this.isWebviewOperational()) {
			return;
		}

		this.clearEnsureRunningTimer();
		this.ensureRunningTimer = setTimeout(() => {
			this.ensureRunningTimer = undefined;
			this.runLifecycleAction('visibility');
		}, 50);
	}

	private clearEnsureRunningTimer(): void {
		this.ensureRunningTimer = this.clearTimer(this.ensureRunningTimer);
	}

	private clearOutputFlushTimer(): void {
		this.outputFlushTimer = this.clearTimer(this.outputFlushTimer);
	}

	private clearTimer(timer: NodeJS.Timeout | undefined): undefined {
		if (timer) {
			clearTimeout(timer);
		}
		return undefined;
	}

	private ensureTerminalRunning(): void {
		if (this.ptyManager.isRunning()) {
			return;
		}
		this.startTerminal();
	}

	private runLifecycleAction(trigger: Trigger, options?: EnsureOptions): void {
		const template = TRIGGER_ACTIONS[trigger];
		const action: LifecycleAction = {
			...template,
			focus: options?.focus ?? template.focus,
		};
		this.applyLifecycleAction(action);
	}

	private applyLifecycleAction(action: LifecycleAction): void {
		if (action.focus) {
			void vscode.commands.executeCommand('snowCliTerminal.focus');
		}

		if (!this.isWebviewOperational()) {
			this.lifecycleQueue.queue(action);
			return;
		}

		this.executeLifecycleAction(this.lifecycleQueue.mergeWithPending(action));
	}

	private executeLifecycleAction(action: LifecycleAction): void {
		if (action.policy === 'restart') {
			this.executeRestart(action);
		} else {
			this.ensureTerminalRunning();
		}

		if (action.requestWebviewFocus) {
			this.requestWebviewFocus();
		}
	}

	private executeRestart(action: LifecycleAction): void {
		this.clearEnsureRunningTimer();
		this.clearFocusRetryTimers();

		if (action.suppressExitBanner && this.terminalSessionNonce > 0) {
			this.suppressedExitSessionNonces.add(this.terminalSessionNonce);
		}

		this.ptyManager.kill();
		if (action.resetFrontend) {
			this.postWebviewMessage({type: 'reset'});
		}
		this.startTerminal();
		this.sendFontConfig();
		this.postWebviewMessage({type: 'fit'});
	}

	private clearTimerSet(timers: Set<NodeJS.Timeout>): void {
		if (timers.size === 0) {
			return;
		}
		for (const timer of timers) {
			clearTimeout(timer);
		}
		timers.clear();
	}

	private clearFocusRetryTimers(): void {
		this.clearTimerSet(this.focusRetryTimers);
	}

	private requestWebviewFocus(): void {
		this.clearFocusRetryTimers();
		if (!this.isWebviewOperational()) {
			return;
		}
		for (const delay of FOCUS_RETRY_DELAYS_MS) {
			const timer = setTimeout(() => {
				this.focusRetryTimers.delete(timer);
				if (!this.isWebviewOperational()) {
					return;
				}
				this.postWebviewMessage({type: 'focus'});
			}, delay);
			this.focusRetryTimers.add(timer);
		}
	}

	private postWebviewMessage(message: ExtensionToWebviewMessage): void {
		if (!this.view || !this.webviewReady) {
			return;
		}
		void this.view.webview.postMessage(message);
	}

	private getExtensionResourceUri(segments: readonly string[]): vscode.Uri {
		return vscode.Uri.joinPath(this.extensionUri, ...segments);
	}

	private getWebviewResourceUri(
		webview: vscode.Webview,
		segments: readonly string[],
	): vscode.Uri {
		return webview.asWebviewUri(this.getExtensionResourceUri(segments));
	}

	private buildScriptTags(
		webview: vscode.Webview,
		scriptSegments: readonly (readonly string[])[],
	): string {
		return scriptSegments
			.map(
				segments =>
					`<script src="${this.getWebviewResourceUri(webview, segments)}"></script>`,
			)
			.join('\n  ');
	}

	private getWebviewAssets(webview: vscode.Webview): {
		xtermCssUri: vscode.Uri;
		sidebarCssUri: vscode.Uri;
		sidebarScriptUri: vscode.Uri;
		scriptTags: string;
	} {
		return {
			xtermCssUri: this.getWebviewResourceUri(webview, XTERM_CSS_SEGMENTS),
			sidebarCssUri: this.getWebviewResourceUri(webview, SIDEBAR_STYLE_SEGMENTS),
			sidebarScriptUri: this.getWebviewResourceUri(webview, SIDEBAR_SCRIPT_SEGMENTS),
			scriptTags: this.buildScriptTags(webview, XTERM_SCRIPT_SEGMENTS),
		};
	}

	private getHtmlForWebview(webview: vscode.Webview): string {
		const cspSource = webview.cspSource;
		const {xtermCssUri, sidebarCssUri, sidebarScriptUri, scriptTags} =
			this.getWebviewAssets(webview);

		return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta http-equiv="Content-Security-Policy"
        content="default-src 'none'; style-src ${cspSource} 'unsafe-inline'; script-src ${cspSource}; font-src ${cspSource};">
  <link rel="stylesheet" href="${xtermCssUri}">
  <link rel="stylesheet" href="${sidebarCssUri}">
</head>
<body>
  <div id="terminal-container"></div>

  ${scriptTags}
  <script src="${sidebarScriptUri}"></script>
</body>
</html>`;
	}

	public sendFilePaths(paths: string[]): void {
		if (paths.length === 0) {
			return;
		}

		this.ensureTerminalRunning();
		if (this.isWebviewOperational()) {
			this.postWebviewMessage({type: 'fileDrop', paths});
			this.requestWebviewFocus();
			return;
		}

		this.writeInputToTerminal(formatPathPayload(paths));
	}

	public dispose(): void {
		this.handleViewDisposed();
	}
}
