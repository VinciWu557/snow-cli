/**
 * 清理终端文本，移除可能被利用的控制字符和 ANSI 转义序列，
 * 防止终端注入攻击。
 */
export function sanitizeTerminalText(text: string): string {
	return text
		.replace(/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/g, '')
		.replace(/\x1B\[[0-9;]*[a-zA-Z]/g, '')
		.replace(/\x1B\][^\x07]*\x07/g, '')
		.replace(/\x1B[^[\]].*/g, '');
}
