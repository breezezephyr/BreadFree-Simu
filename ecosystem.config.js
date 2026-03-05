/**
 * BreadFree PM2 配置
 * 每天 16:00 (东八区) 自动运行每日报告并发送邮件
 *
 * 用法:
 *   pm2 start ecosystem.config.js       # 启动定时任务
 *   pm2 save                             # 保存配置（重启后自动恢复）
 *   pm2 startup                          # 设置开机自启
 *   pm2 list                             # 查看任务状态
 *   pm2 logs breadfree-daily-report      # 查看日志
 *   pm2 stop breadfree-daily-report      # 暂停
 *   pm2 restart breadfree-daily-report   # 重启
 *   pm2 delete breadfree-daily-report    # 删除
 */

module.exports = {
  apps: [
    {
      name: "breadfree-daily-report",
      script: "scripts/daily_report_cron.sh",
      interpreter: "/bin/bash",
      cwd: "/Users/sean.cai/hackthon/BreadFree-Simu",

      // cron 表达式: 每天 16:00 执行（本机时区，需与系统时区一致）
      cron_restart: "0 16 * * 1-5",   // 周一到周五 16:00（A 股交易日）
      // 如需包含周六: "0 16 * * 1-6"
      // 如需每天都跑:  "0 16 * * *"

      // 不作为常驻进程，执行完即退出
      autorestart: false,

      // 日志路径
      out_file: "/Users/sean.cai/hackthon/BreadFree-Simu/logs/pm2-daily-report-out.log",
      error_file: "/Users/sean.cai/hackthon/BreadFree-Simu/logs/pm2-daily-report-err.log",
      log_date_format: "YYYY-MM-DD HH:mm:ss",

      // 环境变量（兜底，.env 会覆盖这里的值）
      env: {
        TZ: "Asia/Shanghai",
      },
    },
  ],
};
