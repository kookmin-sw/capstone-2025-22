package com.example.capstone_2025

import android.content.Intent
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import androidx.appcompat.app.AppCompatActivity
import androidx.core.splashscreen.SplashScreen.Companion.installSplashScreen

// @AndroidEntryPoint
class IntroActivity : AppCompatActivity() {

    // 스플래시 화면이 표시될 시간 (ms)
    private val SPLASH_DISPLAY_LENGTH = 400L

    override fun onCreate(savedInstanceState: Bundle?) {
        // ① 빈 SplashShell 테마 적용
        installSplashScreen()
        super.onCreate(savedInstanceState)
        // ② 커스텀 레이아웃 표시
        setContentView(R.layout.activity_intro)

        // ③ 지정된 시간 후 MainActivity(Flutter) 로 이동
        Handler(Looper.getMainLooper()).postDelayed({
            startActivity(Intent(this, MainActivity::class.java))
            // 2) 들어올 때만 페이드 인, 나갈 때 애니메이션 없음
            overridePendingTransition(R.anim.fade_in, 0)

            finish()
        }, SPLASH_DISPLAY_LENGTH)
    }
}
