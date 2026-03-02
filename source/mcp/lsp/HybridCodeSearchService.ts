import * as path from 'path';
import {ACECodeSearchService} from '../aceCodeSearch.js';
import {LSPManager} from './LSPManager.js';
import {sanitizeTerminalText} from '../../utils/security/sanitizeTerminalText.js';
import type {
	AceProgressCallback,
	AceSearchPhase,
	CodeSymbol,
	CodeReference,
	SemanticSearchResult,
} from '../types/aceCodeSearch.types.js';

export type HybridSearchRuntimeOptions = {
	onProgress?: AceProgressCallback;
	abortSignal?: AbortSignal;
	totalTimeoutMs?: number;
	phaseTimeoutMs?: number;
};

export class HybridCodeSearchService {
	private lspManager: LSPManager;
	private regexSearch: ACECodeSearchService;
	private lspTimeout = 3000; // 3秒超时
	private csharpLspTimeout = 15000; // csharp-ls cold start / solution load can be slow
	private defaultTotalTimeoutMs = 20000;
	private cancelledMessage = 'Search cancelled by user';

	constructor(basePath: string = process.cwd()) {
		this.lspManager = new LSPManager(basePath);
		this.regexSearch = new ACECodeSearchService(basePath);
	}

	private emitProgress(
		options: HybridSearchRuntimeOptions | undefined,
		phase: AceSearchPhase,
		message: string,
		startedAt: number,
		percent?: number,
		metadata?: Record<string, unknown>,
	): void {
		options?.onProgress?.({
			phase,
			message,
			percent,
			elapsedMs: Date.now() - startedAt,
			metadata,
		});
	}

	private createAbortError(message: string = this.cancelledMessage): Error {
		const error = new Error(message);
		error.name = 'AbortError';
		return error;
	}

	private createTimeoutError(message: string): Error {
		const error = new Error(message);
		error.name = 'TimeoutError';
		return error;
	}

	private normalizeSearchError(error: unknown): {
		phase: AceSearchPhase;
		message: string;
	} {
		const message = error instanceof Error ? error.message : String(error);
		const normalizedMessage = message.toLowerCase();

		if (
			(error instanceof Error && error.name === 'AbortError') ||
			normalizedMessage.includes('cancelled') ||
			normalizedMessage.includes('aborted')
		) {
			return {phase: 'cancelled', message: this.cancelledMessage};
		}

		return {phase: 'timeout', message};
	}

	private ensureNotAborted(options?: HybridSearchRuntimeOptions): void {
		if (options?.abortSignal?.aborted) {
			throw this.createAbortError();
		}
	}

	private async runWithTimeout<T>(
		task: () => Promise<T>,
		timeoutMs: number,
		errorMessage: string,
		onHeartbeat?: () => void,
		abortSignal?: AbortSignal,
	): Promise<T> {
		if (abortSignal?.aborted) {
			throw this.createAbortError();
		}

		let heartbeatTimer: NodeJS.Timeout | null = null;
		let timeoutTimer: NodeJS.Timeout | null = null;
		let abortHandler: (() => void) | null = null;
		const guardPromises: Promise<T>[] = [];

		if (Number.isFinite(timeoutMs) && timeoutMs > 0) {
			guardPromises.push(
				new Promise<T>((_, reject) => {
					timeoutTimer = setTimeout(() => {
						reject(this.createTimeoutError(errorMessage));
					}, timeoutMs);
				}),
			);
		}

		if (abortSignal) {
			guardPromises.push(
				new Promise<T>((_, reject) => {
					abortHandler = () => reject(this.createAbortError());
					abortSignal.addEventListener('abort', abortHandler, {once: true});
				}),
			);
		}

		if (onHeartbeat) {
			heartbeatTimer = setInterval(() => {
				if (!abortSignal?.aborted) {
					onHeartbeat();
				}
			}, 4000);
		}

		const taskPromise = task();

		try {
			if (guardPromises.length === 0) {
				return await taskPromise;
			}

			return await Promise.race([taskPromise, ...guardPromises]);
		} finally {
			if (heartbeatTimer) {
				clearInterval(heartbeatTimer);
			}
			if (timeoutTimer) {
				clearTimeout(timeoutTimer);
			}
			if (abortSignal && abortHandler) {
				abortSignal.removeEventListener('abort', abortHandler);
			}
		}
	}

	async findDefinition(
		symbolName: string,
		contextFile?: string,
		line?: number,
		column?: number,
		options?: HybridSearchRuntimeOptions,
	): Promise<CodeSymbol | null> {
		const startedAt = Date.now();
		const totalTimeout =
			typeof options?.totalTimeoutMs === 'number'
				? options.totalTimeoutMs
				: this.defaultTotalTimeoutMs;

		this.emitProgress(
			options,
			'prepare',
			`Preparing definition lookup for \"${sanitizeTerminalText(symbolName)}\"`,
			startedAt,
			5,
		);
		this.ensureNotAborted(options);

		if (contextFile) {
			try {
				this.emitProgress(
					options,
					'search_execute',
					'Trying LSP definition lookup',
					startedAt,
					20,
				);
				const lspResult = await this.findDefinitionWithLSP(
					symbolName,
					contextFile,
					line,
					column,
					options,
				);
				if (lspResult) {
					this.emitProgress(
						options,
						'completed',
						'LSP definition lookup completed',
						startedAt,
						100,
					);
					return lspResult;
				}
			} catch (error) {
				const normalizedError = this.normalizeSearchError(error);
				if (normalizedError.phase === 'cancelled') {
					this.emitProgress(
						options,
						normalizedError.phase,
						normalizedError.message,
						startedAt,
						100,
					);
					return null;
				}

				// LSP 查询失败，降级到 regex 搜索
				this.emitProgress(
					options,
					'fallback',
					'LSP lookup unavailable, switching to index search',
					startedAt,
					45,
				);
			}
		}

		try {
			this.emitProgress(
				options,
				'index_check',
				'Checking symbol index for definition fallback',
				startedAt,
				55,
			);
			this.ensureNotAborted(options);

			const result = await this.runWithTimeout(
				() => this.regexSearch.findDefinition(symbolName, contextFile),
				totalTimeout,
				'Definition search timeout',
				() => {
					this.emitProgress(
						options,
						'search_execute',
						'Searching definition in index...',
						startedAt,
						70,
					);
				},
				options?.abortSignal,
			);

			this.emitProgress(
				options,
				'completed',
				'Definition lookup finished',
				startedAt,
				100,
			);
			return result;
		} catch (error) {
			const normalizedError = this.normalizeSearchError(error);
			this.emitProgress(
				options,
				normalizedError.phase,
				normalizedError.message,
				startedAt,
				100,
			);
			return null;
		}
	}

	private async findDefinitionWithLSP(
		symbolName: string,
		contextFile: string,
		line?: number,
		column?: number,
		options?: HybridSearchRuntimeOptions,
	): Promise<CodeSymbol | null> {
		this.ensureNotAborted(options);
		let position: {line: number; column: number} | null = null;

		const fs = await import('fs/promises');
		const content = await fs.readFile(contextFile, 'utf-8');
		const lines = content.split('\n');
		this.ensureNotAborted(options);

		// 如果提供了 line 和 column，优先使用，但对 C# 需要验证/调整
		// column 使其指向实际的符号 token
		if (line !== undefined && column !== undefined) {
			let adjustedLine = line;
			let adjustedColumn = column;

			if (contextFile.endsWith('.cs')) {
				const tryFindOnLine = (lineIndex: number): number | null => {
					const textLine = lines[lineIndex];
					if (!textLine) return null;
					const symbolRegex = new RegExp(`\\b${symbolName}\\b`);
					const match = symbolRegex.exec(textLine);
					return match ? match.index : null;
				};

				const foundOnSameLine =
					adjustedLine >= 0 && adjustedLine < lines.length
						? tryFindOnLine(adjustedLine)
						: null;
				const foundOnPrevLine =
					foundOnSameLine === null &&
					adjustedLine - 1 >= 0 &&
					adjustedLine - 1 < lines.length
						? tryFindOnLine(adjustedLine - 1)
						: null;

				if (foundOnSameLine !== null) {
					adjustedColumn = foundOnSameLine;
				} else if (foundOnPrevLine !== null) {
					adjustedLine = adjustedLine - 1;
					adjustedColumn = foundOnPrevLine;
				}
			}

			position = {line: adjustedLine, column: adjustedColumn};
		} else {
			// 在 contextFile 中查找符号的首次出现位置
			for (let i = 0; i < lines.length; i++) {
				this.ensureNotAborted(options);
				const textLine = lines[i];
				if (!textLine) continue;

				const symbolRegex = new RegExp(`\\b${symbolName}\\b`);
				const match = symbolRegex.exec(textLine);

				if (match) {
					position = {line: i, column: match.index};
					break;
				}
			}
		}

		if (!position) {
			return null;
		}

		// 向 LSP 请求定义查找（结果可能在其他文件中）
		const timeoutMs =
			typeof options?.phaseTimeoutMs === 'number'
				? options.phaseTimeoutMs
				: contextFile.endsWith('.cs')
				? this.csharpLspTimeout
				: this.lspTimeout;

		this.ensureNotAborted(options);
		let location: Awaited<ReturnType<LSPManager['findDefinition']>>;
		try {
			location = await this.runWithTimeout(
				() =>
					this.lspManager.findDefinition(
						contextFile,
						position.line,
						position.column,
					),
				timeoutMs,
				'Definition search timeout',
				undefined,
				options?.abortSignal,
			);
		} catch (error) {
			const normalizedError = this.normalizeSearchError(error);
			if (normalizedError.phase === 'cancelled') {
				throw this.createAbortError();
			}
			return null;
		}

		this.ensureNotAborted(options);
		if (!location) {
			return null;
		}

		// 将 LSP location 转换为 CodeSymbol
		const filePath = this.uriToPath(location.uri);

		return {
			name: symbolName,
			type: 'function',
			filePath,
			line: location.range.start.line + 1,
			column: location.range.start.character + 1,
			language: this.detectLanguage(filePath),
		};
	}

	async findReferences(
		symbolName: string,
		maxResults = 100,
		options?: HybridSearchRuntimeOptions,
	): Promise<CodeReference[]> {
		const startedAt = Date.now();
		const totalTimeout =
			typeof options?.totalTimeoutMs === 'number'
				? options.totalTimeoutMs
				: this.defaultTotalTimeoutMs;

		this.emitProgress(
			options,
			'prepare',
			`Preparing references lookup for \"${sanitizeTerminalText(symbolName)}\"`,
			startedAt,
			5,
		);
		try {
			this.ensureNotAborted(options);
			this.emitProgress(
				options,
				'search_execute',
				'Searching references in code index',
				startedAt,
				45,
			);
			const refs = await this.runWithTimeout(
				() => this.regexSearch.findReferences(symbolName, maxResults),
				totalTimeout,
				'Reference search timeout',
				() => {
					this.emitProgress(
						options,
						'reference_expand',
						'Expanding symbol references...',
						startedAt,
						70,
					);
				},
				options?.abortSignal,
			);
			this.emitProgress(
				options,
				'completed',
				`Reference lookup completed with ${refs.length} result(s)`,
				startedAt,
				100,
				{count: refs.length},
			);
			return refs;
		} catch (error) {
			const normalizedError = this.normalizeSearchError(error);
			this.emitProgress(
				options,
				normalizedError.phase,
				normalizedError.message,
				startedAt,
				100,
			);
			return [];
		}
	}

	async getFileOutline(
		filePath: string,
		options?: {
			maxResults?: number;
			includeContext?: boolean;
			symbolTypes?: CodeSymbol['type'][];
			onProgress?: AceProgressCallback;
			abortSignal?: AbortSignal;
			timeoutMs?: number;
		},
	): Promise<CodeSymbol[]> {
		const startedAt = Date.now();
		this.emitProgress(
			{onProgress: options?.onProgress},
			'prepare',
			`Preparing file outline for ${filePath}`,
			startedAt,
			5,
		);

		try {
			if (options?.abortSignal?.aborted) {
				throw new Error('Search cancelled by user');
			}
			const timeoutPromise = new Promise<null>(resolve =>
				setTimeout(
					() => resolve(null),
					typeof options?.timeoutMs === 'number'
						? options.timeoutMs
						: this.lspTimeout,
				),
			);

			this.emitProgress(
				{onProgress: options?.onProgress},
				'search_execute',
				'Trying LSP outline provider',
				startedAt,
				35,
			);
			const lspPromise = this.lspManager.getDocumentSymbols(filePath);
			const symbols = await Promise.race([lspPromise, timeoutPromise]);

			if (symbols && symbols.length > 0) {
				this.emitProgress(
					{onProgress: options?.onProgress},
					'completed',
					'LSP outline lookup completed',
					startedAt,
					100,
				);
				return this.convertLSPSymbolsToCodeSymbols(symbols, filePath);
			}
		} catch (error) {
			// LSP 查询失败，降级到 regex
			this.emitProgress(
				{onProgress: options?.onProgress},
				'fallback',
				'LSP outline unavailable, switching to local parser',
				startedAt,
				55,
			);
		}

		const outline = await this.regexSearch.getFileOutline(filePath, {
			maxResults: options?.maxResults,
			includeContext: options?.includeContext,
			symbolTypes: options?.symbolTypes,
		});
		this.emitProgress(
			{onProgress: options?.onProgress},
			'completed',
			`Outline generated with ${outline.length} symbol(s)`,
			startedAt,
			100,
			{count: outline.length},
		);
		return outline;
	}

	private convertLSPSymbolsToCodeSymbols(
		symbols: any[],
		filePath: string,
	): CodeSymbol[] {
		const results: CodeSymbol[] = [];

		const symbolTypeMap: Record<number, CodeSymbol['type']> = {
			5: 'class',
			6: 'method',
			9: 'method',
			10: 'enum',
			11: 'interface',
			12: 'function',
			13: 'variable',
			14: 'constant',
		};

		const processSymbol = (symbol: any) => {
			const range = symbol.location?.range || symbol.range;
			if (!range) return;

			const symbolType = symbolTypeMap[symbol.kind];
			if (!symbolType) return;

			results.push({
				name: symbol.name,
				type: symbolType,
				filePath: this.uriToPath(symbol.location?.uri || filePath),
				line: range.start.line + 1,
				column: range.start.character + 1,
				language: this.detectLanguage(filePath),
			});

			if (symbol.children) {
				for (const child of symbol.children) {
					processSymbol(child);
				}
			}
		};

		for (const symbol of symbols) {
			processSymbol(symbol);
		}

		return results;
	}

	private uriToPath(uri: string): string {
		if (uri.startsWith('file://')) {
			return uri.slice(7);
		}

		return uri;
	}

	private detectLanguage(filePath: string): string {
		const ext = path.extname(filePath).toLowerCase();
		const languageMap: Record<string, string> = {
			'.ts': 'typescript',
			'.tsx': 'typescript',
			'.js': 'javascript',
			'.jsx': 'javascript',
			'.py': 'python',
			'.go': 'go',
			'.rs': 'rust',
			'.java': 'java',
			'.cs': 'csharp',
		};

		return languageMap[ext] || 'unknown';
	}

	async textSearch(
		pattern: string,
		fileGlob?: string,
		isRegex = true,
		maxResults = 100,
		options?: HybridSearchRuntimeOptions,
	) {
		const startedAt = Date.now();
		const totalTimeout =
			typeof options?.totalTimeoutMs === 'number'
				? options.totalTimeoutMs
				: this.defaultTotalTimeoutMs;

		this.emitProgress(
			options,
			'prepare',
			`Preparing text search for pattern: ${sanitizeTerminalText(pattern)}`,
			startedAt,
			5,
		);
		try {
			this.ensureNotAborted(options);
			this.emitProgress(
				options,
				'search_execute',
				'Executing text search',
				startedAt,
				40,
			);
			const results = await this.runWithTimeout(
				() =>
					this.regexSearch.textSearch(pattern, fileGlob, isRegex, maxResults),
				totalTimeout,
				'Text search timeout',
				() => {
					this.emitProgress(
						options,
						'search_execute',
						'Running text search across files...',
						startedAt,
						70,
					);
				},
				options?.abortSignal,
			);
			this.emitProgress(
				options,
				'completed',
				`Text search completed with ${results.length} match(es)`,
				startedAt,
				100,
				{count: results.length},
			);
			return results;
		} catch (error) {
			const normalizedError = this.normalizeSearchError(error);
			this.emitProgress(
				options,
				normalizedError.phase,
				normalizedError.message,
				startedAt,
				100,
			);
			return [];
		}
	}

	async semanticSearch(
		query: string,
		searchType: 'definition' | 'usage' | 'implementation' | 'all' = 'all',
		language?: string,
		symbolType?: CodeSymbol['type'],
		maxResults = 50,
		options?: HybridSearchRuntimeOptions,
	): Promise<SemanticSearchResult> {
		const startedAt = Date.now();
		const totalTimeout =
			typeof options?.totalTimeoutMs === 'number'
				? options.totalTimeoutMs
				: this.defaultTotalTimeoutMs;

		this.emitProgress(
			options,
			'prepare',
			`Preparing semantic search for \"${sanitizeTerminalText(query)}\"`,
			startedAt,
			5,
		);
		this.emitProgress(
			options,
			'index_check',
			'Checking and warming symbol index',
			startedAt,
			25,
		);

		try {
			this.ensureNotAborted(options);
			this.emitProgress(
				options,
				'search_execute',
				'Executing semantic symbol search',
				startedAt,
				50,
			);
			const result = await this.runWithTimeout(
				() =>
					this.regexSearch.semanticSearch(
						query,
						searchType,
						language,
						symbolType,
						maxResults,
					),
				totalTimeout,
				'Semantic search timeout',
				() => {
					this.emitProgress(
						options,
						'reference_expand',
						'Expanding semantic relationships...',
						startedAt,
						72,
					);
				},
				options?.abortSignal,
			);
			this.emitProgress(
				options,
				'result_rank_and_pack',
				'Ranking and packaging semantic results',
				startedAt,
				85,
			);
			this.emitProgress(
				options,
				'completed',
				`Semantic search completed with ${result.totalResults} result(s)`,
				startedAt,
				100,
				{totalResults: result.totalResults},
			);
			return result;
		} catch (error) {
			const normalizedError = this.normalizeSearchError(error);
			this.emitProgress(
				options,
				normalizedError.phase,
				normalizedError.message,
				startedAt,
				100,
			);
			return {
				query,
				symbols: [],
				references: [],
				totalResults: 0,
				searchTime: Date.now() - startedAt,
				degraded: true,
				degradedReason: normalizedError.message,
			};
		}
	}

	async dispose(): Promise<void> {
		await this.lspManager.dispose();
	}
}

export const hybridCodeSearchService = new HybridCodeSearchService();
