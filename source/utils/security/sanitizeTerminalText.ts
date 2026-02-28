const ANSI_ESCAPE_SEQUENCE_REGEX =
	/(?:\u001B\[[0-?]*[ -/]*[@-~])|(?:\u001B\][^\u0007\u001B]*(?:\u0007|\u001B\\))|(?:\u001B[P^_][\s\S]*?\u001B\\)|(?:\u001B[@-_])/g;

const DANGEROUS_CONTROL_CHARS_REGEX =
	/[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F-\u009F]/g;

const BIDI_CONTROL_CHARS_REGEX = /[\u202A-\u202E\u2066-\u2069]/g;

/**
 * 清理终端渲染相关的危险字符，保留可读文本
 */
export function sanitizeTerminalText(text: string | null | undefined): string {
	if (!text) {
		return '';
	}

	return String(text)
		.replace(ANSI_ESCAPE_SEQUENCE_REGEX, '')
		.replace(DANGEROUS_CONTROL_CHARS_REGEX, '')
		.replace(BIDI_CONTROL_CHARS_REGEX, '');
}
