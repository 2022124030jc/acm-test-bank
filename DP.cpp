// 期望DP
// https://atcoder.jp/contests/abc326/tasks/abc326_e

void solve()
{
    int n;
    cin >> n;

    ll in = inv(n, mod);
    ll ans = 0, p = in;
    for (int i = 1; i <= n; i++)
    {
        ll tmp;
        cin >> tmp;

        ans = (ans + p * tmp) % mod;
        p = (p + p * in) % mod;
    }
    cout << ans << endl;
}

//导弹拦截 LIS

void solve()
{
    vector<int> a;
    int x;
    while (cin >> x)
    {
        a.push_back(x);
    }
    int n = a.size();
    vector<int> dp1, dp2;

    for (int i = 0; i < n; i++)
    {
        if (dp1.empty())
        {
            dp1.push_back(a[i]);
        }
        else
        {
            auto it = lower_bound(all(dp1), a[i]);
            if (it == dp1.end())
            {
                dp1.push_back(a[i]);
            }
            else
            {
                *it = a[i];
            }
        }
    }

    for (int i = n - 1; i >= 0; i--)
    {
        if (dp2.empty())
        {
            dp2.push_back(a[i]);
        }
        else
        {
            auto it = upper_bound(all(dp2), a[i]);
            if (it == dp2.end())
            {
                dp2.push_back(a[i]);
            }
            else
            {
                *it = a[i];
            }
        }
    }

    cout << dp2.size() << endl;
    cout << dp1.size() << endl;
}

// 状态DP
//  https://codeforces.com/contest/1842/problem/C

void solve()
{
    int n;
    cin >> n;
    // vector<int> pos(n + 1, 0);
    vector<array<int, 2>> dp(n + 1, {0, 0});
    vector<int> pos(n + 1, 0);

    // memset(dp, 0, sizeof(dp));
    for (int i = 1, tmp; i <= n; i++)
    {
        cin >> tmp;
        dp[i][0] = max(dp[i - 1][1], dp[i - 1][0]);

        if (pos[tmp] != 0)
        {
            dp[i][1] = (i - pos[tmp] + 1) + dp[pos[tmp]][0];
        }
        else
        {
            dp[i][1] = 0;
        }

        if (pos[tmp] == 0 || (dp[pos[tmp]][0] - pos[tmp] < dp[i][0] - i))
        {
            pos[tmp] = i;
        }
    }

    cout << max(dp[n][0], dp[n][1]) << endl;
}

// 树上背包
// http://oj.daimayuan.top/course/8/problem/269
int n, q;
int dp[N][N], siz[N], v[N];
vector<int> g[N];

int dfs(int x)
{
    int s = 1;
    dp[x][1] = v[x];

    for (auto i : g[x])
    {
        int si = dfs(i);
        for (int j = s; j >= 1; j--)
        {
            for (int k = 1; k <= si && j + k <= n; k++)
            {
                dp[x][j + k] = max(dp[x][j + k], dp[x][j] + dp[i][k]);
            }
        }
        s += si;
    }
    return s;
}

void solve()
{
    cin >> n >> q;
    for (int i = 1; i <= n; i++)
    {
        for (int j = 1; j <= n; j++)
        {
            dp[i][j] = -2e9;
        }
    }
    for (int i = 2, f; i <= n; i++)
    {
        cin >> f;
        g[f].push_back(i);
    }
    for (int i = 1; i <= n; i++)
    {
        cin >> v[i];
    }

    dfs(1);
    while (q--)
    {
        int x, y;
        cin >> x >> y;
        cout << dp[x][y] << endl;
    }
}

// 方格取数 两次数字三角形
// https://www.acwing.com/problem/content/1029/

int g[N][N];
int f[2 * N][N][N]; // 枚举i+j相同的格子

void solve()
{
    int n;
    cin >> n;

    int a, b, c;
    while (cin >> a >> b >> c && (a || b || c))
    {
        g[a][b] = c;
    }

    for (int k = 2; k <= 2 * n; k++)
    {
        for (int i1 = 1; i1 < k; i1++)
        {
            for (int i2 = 1; i2 < k; i2++)
            {
                int j1 = k - i1, j2 = k - i2, t = g[i1][j1];
                if (i1 != i2)
                {
                    t += g[i2][j2];
                }
                int &x = f[k][i1][i2];
                x = max(x, f[k - 1][i1][i2] + t);
                x = max(x, f[k - 1][i1 - 1][i2] + t);
                x = max(x, f[k - 1][i1][i2 - 1] + t);
                x = max(x, f[k - 1][i1 - 1][i2 - 1] + t);
            }
        }
    }

    cout << f[2 * n][n][n] << endl;
}

// 最长公共上升子序列
// https://www.acwing.com/problem/content/274/

int a[N], b[N];
int f[N][N]; //

void solve()
{
    int n;
    cin >> n;

    for (int i = 1; i <= n; i++)
    {
        cin >> a[i];
    }
    for (int i = 1; i <= n; i++)
    {
        cin >> b[i];
    }

    for (int i = 1; i <= n; i++)
    {
        int id = 1;
        for (int j = 1; j <= n; j++)
        {
            f[i][j] = f[i - 1][j];

            if (a[i] == b[j])
            {
                f[i][j] = max(f[i][j], id);
            }
            if (b[j] < a[i])
            {
                id = max(id, f[i - 1][j] + 1);
            }
        }
    }

    int ans = 1;
    for (int i = 1; i <= n; i++)
    {
        ans = max(ans, f[n][i]);
    }
    cout << ans << endl;
}

// 贪心+二分求最长上升子序列

int dp[N];
int a[N];

void solve()
{
    int n;
    cin >> n;

    for (int i = 1; i <= n; i++)
    {
        cin >> a[i];
        dp[i] = 2e9;
    }
    dp[0] = 0, dp[1] = a[1];
    for (int i = 2; i <= n; i++)
    {
        int u = lower_bound(dp + 1, dp + n + 1, a[i]) - dp - 1;
        dp[u + 1] = min(dp[u + 1], a[i]);
    }

    for (int i = 1; i < n; i++)
    {
        if (dp[i + 1] == 2e9)
        {
            cout << i << endl;
            break;
        }
    }
}
// 字符串DP
// https://www.luogu.com.cn/problem/P2758

int dp[N][N];

void solve()
{
    string a, b;
    cin >> a >> b;

    int n = a.size(), m = b.size();
    a = " " + a, b = " " + b;

    for (int i = 1; i <= m; i++)
    {
        dp[0][i] = i;
    }
    for (int i = 1; i <= n; i++)
    {
        dp[i][0] = i;
    }

    for (int i = 1; i <= n; i++)
    {
        for (int j = 1; j <= m; j++)
        {
            if (a[i] == b[j])
            {
                dp[i][j] = dp[i - 1][j - 1];
            }
            else
            {
                dp[i][j] = min(min(dp[i - 1][j - 1], dp[i][j - 1]), dp[i - 1][j]) + 1;
            }
        }
    }

    cout << dp[n][m] << endl;
}

// 区间Dp
// P1018

ll dp[N][N];
ll num[N][N]; // i-j的数字总和

void solve()
{
    int n, k;
    cin >> n >> k;
    k++;
    string s;
    cin >> s;
    s = " " + s;
    vector<int> a(n + 1);

    for (int i = 1; i <= n; i++)
    {
        a[i] = s[i] - '0';
    }
    for (int i = 1; i <= n; i++)
    {
        for (int j = i; j <= n; j++)
        {
            num[i][j] = num[i][j - 1] * 10 + a[j];
        }
    }

    for (int i = 1; i < n; i++)
    {
        dp[i][1] = num[1][i];
    }

    for (int i = 2; i <= n; i++)
    {
        for (int l = 2; l <= i; l++) // 枚举分隔点
        {
            for (int r = 2; r <= k; r++)
            {
                dp[i][r] = max(dp[i][r], dp[l - 1][r - 1] * num[l][i]);
            }
        }
    }

    cout << dp[n][k] << endl;
}

// 二进制优化多重背包
void solve()
{
    int n, m;
    cin >> n >> m;

    vector<int> dp(m + 1, -1);
    dp[0] = 0;

    for (int i = 1; i <= n; i++)
    {
        int v, w, s;
        cin >> v >> w >> s;

        int ad = 1;
        while (s)
        {
            if (s < ad)
                ad = s;
            for (int j = m; j - ad * v >= 0; j--)
            {
                dp[j] = max(dp[j], dp[j - ad * v] + ad * w);
            }
            s -= ad, ad *= 2;
        }
    }

    int ans = 0;
    for (int i = 0; i <= m; i++)
        ans = max(ans, dp[i]);

    cout << ans << endl;
}

// 分组背包
void solve()
{
    int n, m;
    cin >> n >> m;

    vector<vector<int>> dp(n + 1, vector<int>(m + 1, -1));
    dp[0][0] = 0;

    for (int i = 1; i <= n; i++)
    {
        int s;
        cin >> s;

        for (int j = 0; j <= m; j++)
        {
            dp[i][j] = dp[i - 1][j];
        }
        while (s--)
        {
            int v, w;
            cin >> v >> w;

            for (int j = m; j - v >= 0; j--)
            {
                if (dp[i - 1][j - v] != -1)
                    dp[i][j] = max(dp[i][j], dp[i - 1][j - v] + w);
            }
        }
    }

    int ans = 0;
    for (int i = 0; i <= m; i++)
        ans = max(ans, dp[n][i]);

    cout << ans << endl;
}

// 最长公共子序列
//  https://www.luogu.com.cn/problem/P1439

void solve()
{
    int n;
    cin >> n;

    vector<int> a(n + 1), b(n + 1);
    vector<vector<int>> dp(n + 1, vector<int>(n + 1));

    for (int i = 1; i <= n; i++)
        cin >> a[i];
    for (int i = 1; i <= n; i++)
        cin >> b[i];

    for (int i = 1; i <= n; i++)
    {
        for (int j = 1; j <= n; j++)
        {
            dp[i][j] = max(max(dp[i - 1][j], dp[i][j - 1]), dp[i - 1][j - 1] + (a[i] == b[j]));
        }
    }

    int ans = 0;
    for (int i = 1; i <= n; i++)
    {
        for (int j = 1; j <= n; j++)
        {
            ans = max(ans, dp[i][j]);
        }
    }

    cout << ans << endl;
}

// 单调决策性
// https://codeforces.com/contest/321/problem/E

ll a[N][N], pre[N][N];
ll dp[N][805];
int n, m;

ll w(int l, int r)
{
    return pre[r][r] - pre[l - 1][r] - pre[r][l - 1] + pre[l - 1][l - 1];
}

void f(int k, int l, int r, int L, int R)
{
    if (l > r)
    {
        return;
    }
    else
    {
        int mid = (l + r) / 2, ma = L;
        for (int i = L; i <= R; i++)
        {
            if (dp[mid][k] > dp[i][k - 1] + w(i + 1, mid))
            {
                dp[mid][k] = dp[i][k - 1] + w(i + 1, mid);
                ma = i;
            }
        }
        f(k, l, mid - 1, L, ma), f(k, mid + 1, r, ma, R);
    }
}

void solve()
{
    cin >> n >> m;

    for (int i = 1; i <= n; i++)
    {
        for (int j = 1; j <= n; j++)
        {
            cin >> pre[i][j];
            pre[i][j] = pre[i][j - 1] + pre[i - 1][j] - pre[i - 1][j - 1] + pre[i][j];
        }
    }

    for (int i = 1; i <= n; i++)
    {
        for (int j = 1; j <= m; j++)
        {
            dp[i][j] = 1e15;
        }
    }
    for (int i = 1; i <= n; i++)
    {
        dp[i][1] = w(1, i);
    }

    for (int k = 2; k <= m; k++)
    {
        f(k, 1, n, 1, n);
    }
    cout << dp[n][m] / 2 << endl;
}

// 概率dp1
//  codeforces.com/problemset/problem/148/D
double dp[N][N];

void solve()
{
    int n, m;
    cin >> n >> m;

    for (int i = 0; i <= n; i++)
        dp[i][0] = 1;
    for (int i = 0; i <= m; i++)
        dp[0][i] = 0;

    for (int i = 1; i <= n; i++)
    {
        for (int j = 1; j <= m; j++)
        {
            double i1 = i, j1 = j;
            dp[i][j] += (i1 / (i1 + j1));

            if (i >= 1 && j >= 2)
            {
                dp[i][j] += ((j1 / (i1 + j1) * ((j1 - 1) / (i1 + j1 - 1)) * ((i1) / (i1 + j1 - 2)))) * dp[i - 1][j - 2];
            }
            if (j >= 3)
            {
                dp[i][j] += ((j1 / (i1 + j1) * ((j1 - 1) / (i1 + j1 - 1)) * ((j1 - 2) / (i1 + j1 - 2)))) * dp[i][j - 3];
            }
        }
    }

    printf("%.10lf\n", dp[n][m]);
}

// 概率dp2
// https://codeforces.com/problemset/problem/768/D

double dp[N * 10][N];

void solve()
{
    int n, q;
    cin >> n >> q;

    dp[1][1] = 1;
    for (int i = 2; i <= 10000; i++)
    {
        for (int j = 1; j <= n; j++)
        {
            double i1 = i, j1 = j;
            dp[i][j] = dp[i - 1][j] * (j1 / n) + dp[i - 1][j - 1] * ((n - j1 + 1) / n);
        }
    }

    while (q--)
    {
        int t;
        cin >> t;

        for (int i = 1; i <= 10000; i++)
        {
            if (dp[i][n] >= (t / 2000.0))
            {
                cout << i << endl;
                break;
            }
        }
    }
}

// hdu7405
// 背包DP 智能车

void solve()
{
    int n, c;
    cin >> n >> c;

    vector<int> dp(c + 1, -1);
    vector<int> a(n);
    for (int i = 0; i < n; i++)
    {
        cin >> a[i];
    }
    sort(all(a));

    int ans = -1;
    dp[0] = 0;
    for (int i = 0; i < n; i++)
    {
        if (ans == -1 && dp[c - a[i]] != -1)
            ans = a[i] - dp[c - a[i]];
        else
            ans = min(ans, a[i] - dp[c - a[i]]);

        if (a[i] == c)
            ans = 0;

        for (int j = c; j - a[i] >= 0; j--)
        {
            if (j - a[i] == 0)
            {
                dp[j] = a[i];
            }
            else
            {
                dp[j] = max(dp[j], dp[j - a[i]]);
            }
        }
    }

    cout << ans << endl;
}

// 二分子序列
#include <bits/stdc++.h>
int main()
{
    int n;
    std::cin >> n;
    std::vector<int> a(n + 1);
    for (int i = 1; i <= n; i++)
    {
        std::cin >> a[i];
    }
    std::vector<int> lis1(n + 5), lis2 = lis1;
    std::vector<int> dp;
    for (int i = 1; i <= n; i++)
    {
        if (dp.empty())
        {
            dp.push_back(a[i]);
        }
        else
        {
            auto it = std::lower_bound(dp.begin(), dp.end(), a[i]);
            if (it == dp.end())
            {
                dp.push_back(a[i]);
            }
            else
            {
                *it = a[i];
            }
        }
        lis1[i] = dp.size();
    }
    dp.clear();
    for (int i = n; i >= 1; i--)
    {
        if (dp.empty())
        {
            dp.push_back({a[i]});
        }
        else
        {
            auto it = lower_bound(dp.begin(), dp.end(), a[i], greater<int>());
            if (it == dp.end())
            {
                dp.push_back(a[i]);
            }
            else
            {
                *it = a[i];
            }
        }
        lis2[i] = dp.size();
    }
    int ans = 0;
    for (int i = 0; i <= n; i++)
    {
        ans = std::max(ans, lis1[i] + lis2[i + 1]);
    }
    std::cout << ans << std::endl;
    return 0;
}

// 背包DP 价值与重量反背
// https: // ac.nowcoder.com/acm/contest/73810/I

void solve()
{
    int n;
    cin >> n;

    vector<ll> dp(N + 1, INF);
    dp[0] = 0;
    for (int i = 1; i <= n; i++)
    {
        ll x, y;
        cin >> x >> y;
        for (int j = N; j - y >= 0; j--)
        {
            dp[j] = min(dp[j], dp[j - y] + x);
        }
    }

    int q;
    cin >> q;

    for (int i = N - 1; i >= 0; i--)
    {
        dp[i] = min(dp[i], dp[i + 1]);
    }
    while (q--)
    {
        ll x;
        cin >> x;

        auto it = upper_bound(all(dp), x) - dp.begin() - 1;
        cout << it << endl;
    }
}

// 虚假的背包 其实是二进制拆位
// https://ac.nowcoder.com/acm/contest/67741/H

void solve()
{
    ll n, m;
    cin >> n >> m;

    vector<int> v(n + 1), w(n + 1);
    for (int i = 1; i <= n; i++)
    {
        cin >> v[i] >> w[i];
    }

    function<ll(int)> get = [&](int x)
    {
        ll res = 0;
        for (int i = 1; i <= n; i++)
        {
            if ((x & w[i]) == w[i])
            {
                res += v[i];
            }
        }
        return res;
    };

    ll ans = get(m);
    for (int i = m; i > 0; i -= lowbit(i))
    {
        ans = max(ans, get(i - 1));
    }
    cout << ans << endl;
}

// https://codeforces.com/contest/1930/problem/D2
// 状态机DP

void solve()
{
    int n;
    cin >> n;

    string s;
    cin >> s;

    ll ans = 0;
    s = " " + s;

    ll dp[n + 1][2];
    for (int i = 0; i <= n; i++)
    {
        dp[i][0] = dp[i][1] = 0;
    }

    if (s[1] == '1')
    {
        dp[1][1] = 1, dp[1][0] = 1e12;
    }
    else
    {
        dp[1][1] = 1, dp[1][0] = 0;
    }
    ans += min(dp[1][0], dp[1][1]);

    for (int i = 2; i <= n; i++)
    {
        if (s[i] == '1')
        {
            dp[i][0] = dp[i - 1][1] + 1;
            dp[i][1] = min(dp[i - 2][0], dp[i - 2][1]) + i;
        }
        else
        {
            dp[i][0] = min(dp[i - 1][0], dp[i - 1][1]);
            dp[i][1] = min(dp[i - 2][0], dp[i - 2][1]) + i;
        }

        ans += min(dp[i][0], dp[i][1]);
    }
    cout << ans << endl;
}

// https://codeforces.com/contest/1932/problem/F
// 线段覆盖问题 优先队列优化dp

void solve()
{
    int n, m;
    cin >> n >> m;

    vector<sb> a(m);
    for (int i = 0; i < m; i++)
    {
        cin >> a[i].l >> a[i].r;
    }
    sort(all(a), cmp);

    vector<int> dp(n + 1);
    priority_queue<array<int, 2>, vector<array<int, 2>>, greater<array<int, 2>>> q1, q2;
    int p = 0, ans = 0, cnt = 0;
    for (int i = 1; i <= n; i++)
    {
        while (p < m && a[p].l == i)
        {
            q1.push({a[p].l, p}), q2.push({a[p].r, p});
            cnt++;
            p++;
        }

        while (!q1.empty() && a[q1.top()[1]].r < i)
        {
            q1.pop();
        }
        while (!q2.empty() && q2.top()[0] < i)
        {
            q2.pop();
            cnt--;
        }

        int x = i;
        if (!q1.empty())
            x = q1.top()[0];
        dp[i] = max(dp[x - 1] + cnt, dp[i - 1]);
        ans = max(ans, dp[x - 1] + cnt);
    }

    cout << ans << endl;
}

// G按照重量排序 两个方向的背包合并
// 南京区域赛2023

struct sb
{
    int w, v;
};

bool cmp(sb x, sb y)
{
    return x.w < y.w;
}

void solve()
{
    int n, m, k;
    cin >> n >> m >> k;

    vector<sb> st(n + 1);
    for (int i = 1; i <= n; i++)
    {
        cin >> st[i].w >> st[i].v;
    }
    sort(st.begin() + 1, st.end(), cmp);

    vector<ll> dp(m + 1), dp2(k + 2);
    vector<ll> l(n + 2), r(n + 2);

    for (int i = 1; i <= n; i++)
    {
        for (int j = m; j - st[i].w >= 0; j--)
        {
            dp[j] = max(dp[j], dp[j - st[i].w] + st[i].v);
        }
        l[i] = dp[m];
    }
    for (int i = n; i >= 1; i--)
    {
        for (int j = k; j - 1 >= 0; j--)
        {
            dp2[j] = max(dp2[j], dp2[j - 1] + st[i].v);
        }
        r[i] = dp2[k];
    }

    ll ans = 0;
    for (int i = 0; i <= n; i++)
    {
        ans = max(ans, l[i] + r[i + 1]);
    }

    cout << ans << endl;
}

// 数位DP windy数

void solve()
{
    int a, b;
    cin >> a >> b;

    function<ll(int)> fun = [&](int x)
    {
        ll ans = 0;
        int len = 0;
        int dp[15][15], num[15];

        memset(dp, -1, sizeof(dp));
        while (x)
        {
            num[++len] = x % 10;
            x /= 10;
        }

        function<int(int, int, bool, bool)> dfs = [&](int pos, int last, bool lead, bool limit)
        {
            int res = 0;
            if (pos == 0)
                return 1;
            if (!lead && !limit && dp[pos][last] != -1)
                return dp[pos][last];

            int up = limit ? num[pos] : 9;

            for (int i = 0; i <= up; i++)
            {
                if (abs(i - last) < 2)
                    continue;
                if (i == 0 && lead)
                    res += dfs(pos - 1, -2, true, limit && i == up);
                else
                    res += dfs(pos - 1, i, false, limit && i == up);
            }

            if (!limit && !lead)
                dp[pos][last] = res; // 更新dp
            return res;
        };

        return dfs(len, -2, true, true);
    };

    cout << fun(b) - fun(a - 1) << endl;
}

// https://atcoder.jp/contests/abc343/tasks/abc343_g
// kmp+状压dp

void solve()
{
    int n;
    cin >> n;

    vector<string> s(n);
    for (int i = 0; i < n; i++)
        cin >> s[i];

    sort(all(s), [&](string &x, string &y)
         { return x.size() < y.size(); });
    vector<string> v;

    function<vector<int>(string)> kmp = [&](string s)
    {
        int n = s.size();
        vector<int> f(n + 1);
        for (int i = 1, j = 0; i < n; i++)
        {
            while (j != 0 && s[i] != s[j])
            {
                j = f[j];
            }
            if (s[i] == s[j])
                j++;
            f[i + 1] = j;
        }

        return f;
    };

    function<bool(string, string)> cxk = [&](string s, string t)
    {
        vector<int> f = kmp(s + "#" + t);
        for (int i = 1; i < f.size(); i++)
        {
            if (f[i] == s.size())
            {
                return true;
            }
        }
        return false;
    };

    for (int i = 0; i < n; i++)
    {
        bool jug = true;
        for (int j = i + 1; j < n; j++)
        {
            if (cxk(s[i], s[j])) // 检查si是否被sj包含
            {
                jug = false;
                break;
            }
        }

        if (jug)
        {
            v.push_back(s[i]);
        }
    }

    s = v;
    n = s.size();
    vector<vector<int>> cost(n + 1, vector<int>(n + 1));
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (i == j)
                continue;
            vector<int> f = kmp(s[i] + "#" + s[j]);
            cost[i][j] = s[j].size() - f.back(); // 计算拼接贡献
        }
    }

    vector<vector<int>> dp(1 << n, vector<int>(n + 1, 1e9));
    for (int i = 0; i < n; i++)
    {
        dp[1 << i][i] = s[i].size();
    }

    // 状压枚举
    for (int s = 1; s < (1 << n); s++)
    {
        for (int i = 0; i < n; i++)
        {
            if ((s >> i) & 1)
            {
                for (int j = 0; j < n; j++)
                {
                    if ((s >> j & 1) != 1)
                    {
                        dp[s | 1 << j][j] = min(dp[s | 1 << j][j], dp[s][i] + cost[i][j]);
                    }
                }
            }
        }
    }

    int ans = dp[(1 << n) - 1][0];
    for (int i = 1; i < n; i++)
    {
        ans = min(ans, dp[(1 << n) - 1][i]);
    }

    cout << ans << endl;
}

// https://codeforces.com/contest/1632/problem/D
// DS+二分+贪心
// 线段树+二分或者双指针预处理出线段 将其转换成线段覆盖的问题

int tr[N * 4];
int a[N];

void push_up(int p)
{
    tr[p] = gcd(tr[p * 2], tr[p * 2 + 1]);
}

void build(int p, int l, int r)
{
    if (l == r)
    {
        tr[p] = a[l];
        return;
    }

    int mid = (l + r) / 2;
    build(p * 2, l, mid);
    build(p * 2 + 1, mid + 1, r);

    push_up(p);
}

int query(int p, int l, int r, int pl, int pr)
{
    if (l >= pl && r <= pr)
    {
        return tr[p];
    }

    int mid = (l + r) / 2;
    if (pr <= mid)
    {
        return query(p * 2, l, mid, pl, pr);
    }
    else if (pl >= mid + 1)
    {
        return query(p * 2 + 1, mid + 1, r, pl, pr);
    }
    else
    {
        return gcd(query(p * 2, l, mid, pl, pr), query(p * 2 + 1, mid + 1, r, pl, pr));
    }
}

void solve()
{
    int n;
    cin >> n;

    for (int i = 1; i <= n; i++)
    {
        cin >> a[i];
    }
    build(1, 1, n);

    vector<sb> v;
    // int pl = 1;
    // for (int pr = 1; pr <= n; pr++)
    // {
    //     while (pl < pr && query(1, 1, n, pl+1, pr) <= (pr - pl))
    //         pl++;
    //     if (query(1, 1, n, pl, pr) == (pr - pl + 1))
    //     {
    //         v.push_back({pl, pr});
    //         pl = pr;
    //     }
    // }

    function<bool(int, int)> cxk = [&](int mid, int i)
    {
        if (query(1, 1, n, mid, i) <= (i - mid + 1))
        {
            return true;
        }
        else
        {
            return false;
        }
    };
    for (int i = 1; i <= n; i++)
    {
        int l = 1, r = i;

        while (l < r)
        {
            int mid = (l + r + 1) / 2;

            if (cxk(mid, i))
            {
                l = mid;
            }
            else
            {
                r = mid - 1;
            }
        }

        if (query(1, 1, n, l, i) == (i - l + 1))
        {
            v.push_back({l, i});
        }
    }

    // for (auto [x, y] : v)
    //     cout << x << " " << y << endl;

    // vector<int> dp(n+1);
    // priority_queue<int,vector<int>,greater<int>> q;
    sort(all(v), cmp);
    int p = 0, pre = 0, now = 0;
    for (int i = 1; i <= n; i++)
    {
        int l = 0;
        while (p < v.size() && i == v[p].r)
        {
            // q.push(v[p].l);
            l = max(l, v[p].l);
            p++;
        }

        // while(!q.empty() && q.top()<i)
        // {

        // }

        if (l > pre)
        {
            pre = i;
            now++;
        }

        cout << now << " \n"[i == n];
    }
}

// http://oj.daimayuan.top/course/8/problem/368
// 状压dp 子集和

void solve()
{
    unsigned n, a, b, c;
    cin >> n >> a >> b >> c;

    function<unsigned()> fun = [&]()
    {
        a ^= a << 16;
        a ^= a >> 5;
        a ^= a << 1;
        ll t = a;
        a = b;
        b = c;
        c ^= t ^ a;
        return c;
    };

    vector<ll> f(1 << n), g(1 << n);
    for (int i = 0; i < (1 << n); i++)
    {
        f[i] = fun();
    }

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < (1 << n); j++)
        {
            if ((1 << i) & j)
            {
                f[j] += f[j - (1 << i)];
            }
        }
    }

    ll ans = 0;
    for (int i = 0; i < (1 << n); i++)
    {
        ans ^= f[i];
    }

    cout << ans << endl;
}

// http://oj.daimayuan.top/course/8/problem/369
// 利用子集状压

// https://codeforces.com/problemset/problem/1475/G
// 刷表或根号

void solve()
{
    int n;
    cin >> n;

    vector<int> a(n);
    vector<int> dp(N);
    for (auto &i : a)
        cin >> i;

    sort(all(a));
    int ans = 0, pre = 0;
    for (auto i : a)
    {
        dp[i]++;
        if (i == pre)
            continue;
        pre = i;
        int j = 1;
        while ((j + 1) * (j + 1) <= i)
            j++;
        for (; j >= 1; j--)
        {
            if (j >= i)
                continue;
            if (i % j == 0 && dp[j] != 0)
            {
                dp[i] = max(dp[i], dp[j] + 1);
            }
        }
        for (j = 2; j * j <= i; j++)
        {
            if (i % j == 0 && dp[i / j] != 0)
            {
                dp[i] = max(dp[i], dp[i / j] + 1);
            }
        }
    }

    for (int i = 1; i <= 2e5; i++)
    {
        ans = max(ans, dp[i]);
    }

    cout << n - ans << endl;
}

// 刷表
void solve()
{
    int n;
    cin >> n;

    vector<int> a(n), dp(N), cnt(N);
    set<int> se;
    for (auto &i : a)
    {
        cin >> i;
        cnt[i]++;
        se.insert(i);
    }

    int ans = 0;
    vector<int> v;
    for (auto i : se)
        v.push_back(i);

    for (auto i : v)
    {
        for (int j = 2 * i; j <= N; j += i)
        {
            dp[j] = max(dp[j], dp[i] + cnt[i]);
        }

        ans = max(ans, dp[i] + cnt[i]);
    }

    cout << n - ans << endl;
}

// https://ac.nowcoder.com/acm/contest/76681/J
// 状压+期望

void solve()
{
    int n, m;

    cin >> n >> m;
    vector<double> a(n + 1);
    vector<bool> vis(1 << n + 10);
    vector<double> dp(1 << n + 10, -1);

    for (int i = 1; i <= n; i++)
    {
        cin >> a[i];
    }

    function<bool(int)> cxk = [&](int x)
    {
        ll res = 0;
        for (int i = 0; i < n; i++)
        {
            if ((x >> i) & 1)
            {
                res += a[i + 1];
            }
            if (res >= m)
                return true;
        }
        return false;
    };

    for (int i = 0; i < (1 << n); i++)
    {
        vis[i] = cxk(i);
    }

    auto dfs = [&](auto &&self, int x) -> double
    {
        if (vis[x])
            return 0.0;
        if (dp[x] != -1)
            return dp[x];

        dp[x] = n;
        for (int i = 0; i < n; i++)
        {
            if (!((x >> i) & 1))
            {
                dp[x] += self(self, x | (1 << i));
            }
        }

        dp[x] /= (n - __builtin_popcount(x));
        return dp[x];
    };

    cout << fixed << setprecision(6);
    cout << dfs(dfs, 0) << endl;
}

// https://codeforces.com/contest/1083/problem/A
// 树D 题意：选择一条路径 使得路径上的所有点权和减去边权和最大

void solve()
{
    int n;
    cin >> n;

    vector<ll> w(n + 1);
    vector<vector<array<ll, 2>>> g(n + 1, vector<array<ll, 2>>());
    ll ans = 0;
    for (int i = 1; i <= n; i++)
    {
        cin >> w[i];
        ans = max(ans, w[i]);
    }

    for (int i = 1; i < n; i++)
    {
        ll x, y, v;
        cin >> x >> y >> v;

        g[x].push_back({y, v});
        g[y].push_back({x, v});
    }

    vector<ll> dp(n + 1);
    function<void(int, int, ll)> dfs = [&](int x, int pre, ll sum)
    {
        if (g[x].size() == 1 && x != 1)
        {
            dp[x] = w[x];
            return;
        }

        for (auto [i, v] : g[x])
        {
            if (i == pre)
                continue;
            dfs(i, x, max(sum - v + w[x], 0ll));
        }

        vector<ll> vc;
        vc.push_back(0), vc.push_back(sum);
        for (auto [i, v] : g[x])
        {
            dp[x] = max(dp[x], dp[i] - v);
            vc.push_back(dp[i] - v);
        }
        dp[x] += w[x];
        sort(all(vc), greater<ll>());

        ans = max(ans, vc[0] + vc[1] + w[x]);
    };

    dfs(1, 0, 0ll);

    cout << ans << endl;
}

// acwing 斜率优化模板题 n^2->双指针->二分斜率

void solve()
{
    int n;
    cin >> n;

    vector<ll> dp(n + 1, 1e16);
    ll s;
    cin >> s;

    vector<ll> t(n + 1), c(n + 1), pre_t(n + 1), pre_c(n + 1);
    for (int i = 1; i <= n; i++)
    {
        cin >> t[i] >> c[i];
        pre_t[i] = pre_t[i - 1] + t[i];
        pre_c[i] = pre_c[i - 1] + c[i];
    }

    // dp[0] = 0;
    // 朴素n^2算法
    // for(int i = 1;i<=n;i++)
    // {
    //     for(int j = 0;j<i;j++)
    //     {
    //         dp[i] = min(dp[i],dp[j]+s*(pre_c[n]-pre_c[j])+(pre_c[i]-pre_c[j])*pre_t[i]);
    //     }
    // }

    // 斜率优化
    // f[i] = f[j]+s*(sumc[n]-sumc[j])+sumt[i]*(sumc[i]-sumc[j])
    // 移项处理 f[j]和sumc[j]为变量
    // f[i] = f[j]-(s+sumt[i])*sumc[j]+s*sumc[n]+sumt[i]*sumc[i]
    // f[j] = (s+sumt[i])*sumc[j]+(f[i]-s*sumc[n]-sumt[i]*sumc[i])
    // 可以看成直线方程 形如y = k*x+b
    // 那么 斜率为定值,f[i]属于截距
    // 维护一个凸包

    deque<int> dq;
    dq.push_front(0);
    dp[0] = 0;
    dp[1] = dp[0] + s * (pre_c[n] - pre_c[0]) + (pre_c[1] - pre_c[0]) * pre_t[1];
    dq.push_back(1);
    for (int i = 2; i <= n; i++)
    {
        // while (dq.size() > 1 && (dp[dq[1]] - dp[dq[0]]) < (s + pre_t[i]) * (pre_c[dq[1]] - pre_c[dq[0]]))
        // {
        //     dq.pop_front();
        // }
        // 删掉前面的点，斜率单调递增
        int l = 0, r = dq.size() - 2;
        while (l < r)
        {
            int mid = (l + r) / 2;
            if ((dp[dq[mid + 1]] - dp[dq[mid]]) < (s + pre_t[i]) * (pre_c[dq[mid + 1]] - pre_c[dq[mid]]))
            {
                l = mid + 1;
            }
            else
            {
                r = mid;
            }
        }

        dp[i] = dp[dq[l]] + s * (pre_c[n] - pre_c[dq[l]]) + (pre_c[i] - pre_c[dq[l]]) * pre_t[i];

        while (dq.size() > 1)
        {
            int x1 = dq[dq.size() - 1], x2 = dq[dq.size() - 2];

            if ((pre_c[x1] - pre_c[x2]) * (dp[i] - dp[x2]) < (pre_c[i] - pre_c[x2]) * (dp[x1] - dp[x2]))
            {
                dq.pop_back();
            }
            else
            {
                break;
            }
        }

        if (dq.size() == 1)
        {
            dq.push_back(i);
            continue;
        }
        int x1 = dq[dq.size() - 1], x2 = dq[dq.size() - 2];
        if ((pre_c[x1] - pre_c[x2]) * (dp[i] - dp[x2]) < (pre_c[i] - pre_c[x2]) * (dp[x1] - dp[x2]))
            dq.push_back(i);
    }

    cout << dp[n] << endl;
}

void solve()
{
    int n;
    cin >> n;

    vector<ll> dp(n + 2);
    ll s;
    cin >> s;

    vector<ll> t(n + 2), c(n + 2), pre_t(n + 2), pre_c(n + 2), dq(n + 2);
    int tt = 0;
    for (int i = 1; i <= n; i++)
    {
        cin >> t[i] >> c[i];
        pre_t[i] = pre_t[i - 1] + t[i];
        pre_c[i] = pre_c[i - 1] + c[i];
    }

    for (int i = 1; i <= n; i++)
    {
        int l = 0, r = tt;

        while (l < r)
        {
            int mid = (l + r) / 2;
            if ((dp[dq[mid + 1]] - dp[dq[mid]]) > (s + pre_t[i]) * (pre_c[dq[mid + 1]] - pre_c[dq[mid]]))
            {
                r = mid;
            }
            else
            {
                l = mid + 1;
            }
        }

        dp[i] = dp[dq[l]] + s * (pre_c[n] - pre_c[dq[l]]) + (pre_c[i] - pre_c[dq[l]]) * pre_t[i];

        while (tt >= 1)
        {
            int x1 = dq[tt], x2 = dq[tt - 1];

            if ((double)(pre_c[x1] - pre_c[x2]) * (dp[i] - dp[x1]) <= (double)(pre_c[i] - pre_c[x1]) * (dp[x1] - dp[x2]))
            {
                tt--;
            }
            else
            {
                break;
            }
        }

        dq[++tt] = i;
    }

    cout << dp[n] << endl;
}

//https: // codeforces.com/contest/1066/problem/F
//状态机dp

void solve()
{
    int n;
    cin >> n;

    vector<sb> a(n + 1);
    vector<vector<ll>> dp(n + 10, vector<ll>(2)), id(n + 10, vector<ll>(2));
    for (int i = 1; i <= n; i++)
    {
        cin >> a[i].x >> a[i].y;
    }
    sort(a.begin() + 1, a.end(), cmp);
    int l = 0, r = 0; // 记录最左边跟最下面

    if (n == 1)
    {
        cout << a[1].x + a[1].y << endl;
        return;
    }

    function<ll(int, int)> cal = [&](int s1, int s2)
    {
        return abs(a[s1].x - a[s2].x) + abs(a[s1].y - a[s2].y);
    };

    int p = 0;
    for (int i = 1; i <= n; i++)
    {
        if (max(a[i].x, a[i].y) > max(a[i - 1].x, a[i - 1].y))
        {
            p++;
            dp[p][0] = min(dp[p - 1][0] + cal(id[p - 1][0], l), dp[p - 1][1] + cal(id[p - 1][1], l));
            dp[p][1] = min(dp[p - 1][0] + cal(id[p - 1][0], r), dp[p - 1][1] + cal(id[p - 1][1], r));

            ll xx = dp[p][0], yy = dp[p][1];
            dp[p][0] = min(yy + cal(l, r), xx + 2 * cal(l, r));
            dp[p][1] = min(xx + cal(l, r), yy + 2 * cal(l, r));
            id[p][0] = l, id[p][1] = r;
            l = r = i;
        }
        else
        {
            if (a[i].x < a[l].x || (a[i].x == a[l].x && a[i].y > a[l].y))
                l = i;
            if (a[i].y < a[r].y || (a[i].y == a[r].y && a[i].x > a[r].x))
                r = i;
        }
    }

    if (n >= 1)
    {
        p++;
        dp[p][0] = min(dp[p - 1][0] + cal(id[p - 1][0], l), dp[p - 1][1] + cal(id[p - 1][1], l));
        dp[p][1] = min(dp[p - 1][0] + cal(id[p - 1][0], r), dp[p - 1][1] + cal(id[p - 1][1], r));

        ll xx = dp[p][0], yy = dp[p][1];
        dp[p][0] = min(yy + cal(l, r), xx + 2 * cal(l, r));
        dp[p][1] = min(xx + cal(l, r), yy + 2 * cal(l, r));
        l = r = n;
    }

    cout << min(dp[p][0], dp[p][1]) << endl;
}

// https://www.luogu.com.cn/problem/P3959
//妙妙状压，存树的最大高度，考虑集合之间的转移

void solve()
{
    int n, m;
    cin >> n >> m;

    vector<vector<ll>> g(n + 1, vector<ll>(n + 1, INF));
    vector<ll> s(1 << n);
    for (int i = 0; i < n; i++)
    {
        g[i][i] = 0;
    }
    while (m--)
    {
        ll x, y, v;
        cin >> x >> y >> v;
        x--, y--;

        g[x][y] = g[y][x] = min(g[x][y], v);
    }

    for (int i = 0; i < (1 << n); i++)
    {
        for (int j = 0; j < n; j++)
        {
            if ((i >> j) & 1)
            {
                for (int k = 0; k < n; k++)
                {
                    if (g[j][k] != INF)
                    {
                        s[i] |= (1 << k);
                    }
                }
            }
        }
    }

    vector<vector<ll>> dp(1 << n, vector<ll>(n + 1, INF)); // 状态，树的最大高度
    for (int i = 0; i < n; i++)
    {
        dp[1 << i][0] = 0;
    }

    for (int i = 0; i < (1 << n); i++)
    {
        for (int j = i - 1; j > 0; j = (j - 1) & i)
        {
            if ((s[j] & i) == i)
            {
                int ad = i ^ j;
                ll cost = 0;
                for (int k = 0; k < n; k++)
                {
                    ll tmp = INF;
                    if ((ad >> k) & 1)
                    {
                        for (int l = 0; l < n; l++)
                        {
                            if ((j >> l) & 1)
                            {
                                tmp = min(tmp, g[k][l]);
                            }
                        }
                        cost += tmp;
                    }
                }

                for (int k = 1; k < n; k++)
                {
                    dp[i][k] = min(dp[i][k], dp[j][k - 1] + cost * k);
                }
            }
        }
    }

    ll ans = INF;
    for (int i = 0; i < n; i++)
    {
        ans = min(ans, dp[(1 << n) - 1][i]);
    }

    cout << ans << endl;
}

// https://codeforces.com/contest/1942/problem/D
//指针优化DP转移

void solve()
{
    int n, k;
    cin >> n >> k;

    vector<vector<ll>> a(n + 2, vector<ll>(n + 2)), dp(n + 2, vector<ll>());
    for (int i = 1; i <= n; i++)
    {
        for (int j = i; j <= n; j++)
        {
            cin >> a[i][j];
        }
    }

    dp[0].push_back(0);

    for (int i = 1; i <= n + 1; i++)
    {
        priority_queue<array<ll, 3>> q;
        for (int j = 0; j <= i - 1; j++)
        {
            q.push({dp[j][0] + a[j + 1][i - 1], j, 0});
        }

        while (q.size() && dp[i].size() != k)
        {
            auto [v, x, y] = q.top();
            q.pop();

            dp[i].push_back(v);
            if (y + 1 < dp[x].size())
                q.push({dp[x][y + 1] + a[x + 1][i - 1], x, y + 1});
        }
    }

    for (auto i : dp[n + 1])
        cout << i << " ";
    cout << endl;
}

// https://www.luogu.com.cn/problem/P2048
//三角形转移DP ST表分裂

void solve()
{
    int n, k, L, R;
    cin >> n >> k >> L >> R;

    vector<ll> a(n + 2), pre(n + 2), lg(n + 2);
    lg[0] = -1;

    for (int i = 1; i <= n; i++)
    {
        lg[i] = lg[i / 2] + 1;
    }

    for (int i = 1; i <= n; i++)
    {
        cin >> a[i];
        pre[i] = a[i] + pre[i - 1];
    }

    vector<vector<ll>> st(n + 2, vector<ll>(32)), pos(n + 2, vector<ll>(32));
    for (int i = 1; i <= n; i++)
    {
        st[i][0] = pre[i];
        pos[i][0] = i;
    }

    for (int j = 1; j <= 20; j++)
    {
        for (int i = 1; i + (1 << j) - 1 <= n; i++)
        {
            if (st[i][j - 1] >= st[i + (1 << (j - 1))][j - 1])
            {
                st[i][j] = st[i][j - 1];
                pos[i][j] = pos[i][j - 1];
            }
            else
            {
                st[i][j] = st[i + (1 << (j - 1))][j - 1];
                pos[i][j] = pos[i + (1 << (j - 1))][j - 1];
            }
        }
    }

    priority_queue<array<ll, 5>> q; // 贡献 最佳点 起点 pl pr

    function<void(int, int, int)> add = [&](int x, int pl, int pr)
    {
        ll v, t;
        int m = lg[pr - pl + 1];

        if (pl == pr)
        {
            q.push({pre[pl] - pre[x - 1], pl, x, pl, pr});
            return;
        }
        if (st[pl][m] >= st[pr - (1 << m) + 1][m])
        {
            v = st[pl][m];
            t = pos[pl][m];
        }
        else
        {
            v = st[pr - (1 << m) + 1][m];
            t = pos[pr - (1 << m) + 1][m];
        }

        q.push({v - pre[x - 1], t, x, pl, pr});
    };

    for (int i = 1; i + L - 1 <= n; i++)
    {
        int r = min(n, i + R - 1);
        add(i, i + L - 1, r);
    }

    ll ans = 0;
    while (k--)
    {
        auto [v, id, x, pl, pr] = q.top();
        q.pop();
        ans += v;

        if (id - 1 >= pl)
        {
            add(x, pl, id - 1);
        }
        if (id + 1 <= pr)
        {
            add(x, id + 1, pr);
        }
    }

    cout << ans << endl;
}

// https://www.luogu.com.cn/problem/P5283
//01trie+三角形转移DP

ll tr[N * 40][2], siz[N * 40], tot;

void insert(ll x)
{
    int p = 0;
    for (int i = 39; i >= 0; i--)
    {
        int u = (x >> i) & 1;
        if (!tr[p][u])
        {
            tr[p][u] = ++tot;
        }
        p = tr[p][u];
        siz[p]++;
    }
}

ll ask(ll x, ll k) // 查找与x异或的第k大
{
    int p = 0;
    ll res = 0;
    for (ll i = 39; i >= 0; i--)
    {
        int u = (x >> i) & 1;
        if (siz[tr[p][u ^ 1]] >= k)
        {
            p = tr[p][u ^ 1];
            u ^= 1;
            res += ((u & 1ll) << i);
        }
        else
        {
            k -= siz[tr[p][u ^ 1]];
            p = tr[p][u];
            res += ((u & 1ll) << i);
        }
    }
    return res;
}

void solve()
{
    ll n, k;
    cin >> n >> k;

    vector<ll> a(n + 1), pre(n + 1);
    insert(0);
    for (int i = 1; i <= n; i++)
    {
        cin >> a[i];
        pre[i] = pre[i - 1] ^ a[i];
        insert(pre[i]);
    }

    priority_queue<array<ll, 3>> q;
    for (int i = 0; i <= n; i++)
    {
        q.push({ask(pre[i], 1) ^ pre[i], i, 1});
    }

    ll ans = 0;
    k *= 2;
    while (k--)
    {
        auto [v, x, y] = q.top();
        q.pop();

        ans += v;
        if (y + 1 <= n + 1)
        {
            q.push({ask(pre[x], y + 1) ^ pre[x], x, y + 1});
        }
    }

    cout << ans / 2 << endl;
}

// https://codeforces.com/contest/1918/problem/D
//二分+dp

void solve()
{
    int n;
    cin >> n;

    vector<ll> a(n + 1);
    for (int i = 1; i <= n; i++)
    {
        cin >> a[i];
    }

    ll l = 1, r = 1e14;

    auto cxk = [&](ll mid) -> bool
    {
        vector<ll> dp(n + 1); // 表示以i为分割点的最小分割出来的和
        ll p = 0, now = 0;
        deque<array<ll, 2>> dq; // val id
        dq.push_back({0, 0});

        for (int i = 1; i <= n; i++)
        {
            if (a[i] > mid)
                return false;
            while (dq.front()[1] + 1 < p)
                dq.pop_front();
            dp[i] = dq.front()[0] + a[i];
            now += a[i];
            while (now > mid)
            {
                now -= a[p++];
            }
            while (!dq.empty() && dq.back()[0] > dp[i])
                dq.pop_back();
            dq.push_back({dp[i], i});
        }

        now = 0;
        for (int i = n; i >= 1; i--)
        {
            if (now > mid)
                break;
            dp[n] = min(dp[n], dp[i]);
            now += a[i];
        }

        if (dp[n] <= mid)
            return true;
        else
            return false;
    };

    while (l < r)
    {
        ll mid = (l + r) / 2;
        if (cxk(mid))
            r = mid;
        else
            l = mid + 1;
    }
    cout << r << endl;
}

// https://atcoder.jp/contests/abc348/tasks/abc348_e
// 换根dp

void solve()
{
    int n;
    cin >> n;

    vector<vector<int>> g(n + 1, vector<int>());
    vector<ll> siz(n + 1); // 子树权值和
    vector<ll> a(n + 1);   // 点权

    for (int i = 1; i < n; i++)
    {
        int x, y;
        cin >> x >> y;
        g[x].push_back(y), g[y].push_back(x);
    }

    for (int i = 1; i <= n; i++)
    {
        cin >> a[i];
    }

    ll ans = 0;
    function<void(int, int)> dfs1 = [&](int x, int pre) // 预处理siz,以及以1为根节点时的ans
    {
        ll res = a[x];
        siz[x] = a[x];

        for (auto i : g[x])
        {
            if (i == pre)
                continue;
            dfs1(i, x);
            siz[x] += siz[i];
        }
        ans += siz[x];
    };

    function<void(ll, ll, ll)> dfs2 = [&](ll x, ll pre, ll sum)
    {
        ans = min(ans, sum);

        for (auto i : g[x])
        {
            if (i == pre)
                continue;
            dfs2(i, x, sum - siz[i] + (siz[1] - siz[i]));
        }
    };

    dfs1(1, 0);
    ans -= siz[1];
    dfs2(1, 0, ans); // 换根

    cout << ans << endl;
}

// https://codeforces.com/contest/1955/problem/H
// 状压dp 解决重复问题
int p[14];

void solve()
{
    int n, m, k;
    cin >> n >> m >> k;

    vector<vector<char>> c(n + 1, vector<char>(m + 1));
    for (int i = 1; i <= n; i++)
    {
        for (int j = 1; j <= m; j++)
        {
            cin >> c[i][j];
        }
    }

    vector<vector<int>> val(k + 1, vector<int>(13));

    for (int i = 1; i <= k; i++)
    {
        int x, y, v;
        cin >> x >> y >> v;

        for (int r = 0; r <= 12; r++)
        {
            int tmp = 0;
            for (int i1 = 1; i1 <= n; i1++)
            {
                for (int j1 = 1; j1 <= m; j1++)
                {
                    if (c[i1][j1] == '#' && (i1 - x) * (i1 - x) + (j1 - y) * (j1 - y) <= r * r)
                    {
                        tmp += v;
                    }
                }
            }

            val[i][r] = max(0, tmp - p[r]);
        }
    }

    vector<vector<int>> dp(k + 1, vector<int>(1 << 13));

    int ans = 0;

    for (int i = 1; i <= k; i++)
    {
        for (int j = 0; j < (1 << 12); j++)
        {
            dp[i][j] = dp[i - 1][j];
            for (int r = 1; r <= 12; r++)
            {
                if (j & (1 << (r - 1)))
                {
                    dp[i][j] = max(dp[i][j], dp[i - 1][j ^ ((1 << (r - 1)))] + val[i][r]);
                }
            }
            ans = max(ans, dp[i][j]);
        }
    }

    cout << ans << endl;
}

// https://codeforces.com/contest/1950/problem/G
//状压DP

void solve()
{
    int n;
    cin >> n;
      vector<array<string, 2>> a(n + 1);
    vector<vector<int>> b(n + 1, vector<int>(n + 1));
    map<string, vector<int>> mp1, mp2;
      int ans = 0;
    for (int i = 0; i < n; i++)
    {
        cin >> a[i][0] >> a[i][1];
        for (auto j : mp1[a[i][0]])
        {
            b[i][j] = b[j][i] = 1;
        }
        for (auto j : mp2[a[i][1]])
        {
            b[i][j] = b[j][i] = 1;
        }
        mp1[a[i][0]].push_back(i);
        mp2[a[i][1]].push_back(i);
    }

    vector<vector<int>> dp(1 << n, vector<int>(n));
    for (int i = 0; i < n; i++)
    {
        dp[1 << i][i] = 1;
    }

    for (int i = 0; i < (1 << n); i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (i & (1 << j))
            {
                for (int k = 0; k < n; k++)
                {
                    if (k == j)
                        continue;
                    if ((i & (1 << k)) && b[k][j])
                    {
                        dp[i][j] = max(dp[i][j], dp[i - (1 << j)][k]);
                    }
                }
                if (dp[i][j])
                    ans = max(ans, __builtin_popcount(i));
            }
        }
    }
     cout << n - ans << endl;
}

// 天梯赛-关于深度优先搜索和逆序对的题应该不会很难吧这件事
// 树D + 树状数组+计数

ll d[N];
int n;

void add(int x, ll v)
{
    while (x <= n)
    {
        d[x] += v;
        x += lowbit(x);
    }
}

ll ask(int x)
{
    ll res = 0;
    while (x > 0)
    {
        res += d[x];
        x -= lowbit(x);
    }

    return res;
}

ll p[N + 10]; // 预处理阶乘
void init()
{
    p[1] = 1;
    for (int i = 2; i <= N; i++)
    {
        p[i] = (p[i - 1] * i) % mod;
    }
}

void solve()
{
    int r;
    cin >> n >> r;

    vector<vector<int>> g(n + 1, vector<int>());
    for (int i = 1; i < n; i++)
    {
        int x, y;
        cin >> x >> y;
        g[x].push_back(y);
        g[y].push_back(x);
    }

    vector<ll> siz(n + 1);
    ll sum = 1; // 所有可能性
    ll ans = 0; // 逆序对数量+非同链对数
    function<void(int, int)> dfs = [&](int x, int pre)
    {
        ll now = 0;
        ans += (ask(n) - ask(x));
        ans %= mod;
        add(x, 1);
        for (auto i : g[x])
        {
            if (i == pre)
                continue;
            dfs(i, x);
            ans = (ans + siz[x] * siz[i] % mod * inv(2, mod) % mod) % mod;
            siz[x] = (siz[x] + siz[i]) % mod;
            now++;
        }

        sum = (sum * max(1ll, p[now])) % mod;
        siz[x]++;
        add(x, -1);
    };

    dfs(r, 0);
    cout << (ans * sum) % mod << endl;
}

// https://codeforces.com/contest/1156/problem/D
// 树D

void solve()
{
    int n;
    cin >> n;

    vector<vector<array<int, 2>>> g(n + 1, vector<array<int, 2>>());

    for (int i = 1; i < n; i++)
    {
        int x, y, id;
        cin >> x >> y >> id;

        g[x].push_back({y, id});
        g[y].push_back({x, id});
    }

    vector<ll> cn1(n + 1), cn2(n + 1), cn3(n + 1), cn4(n + 1);
    ll ans = 0, x1 = 0;
    function<void(int, int)> dfs = [&](int x, int pre)
    {
        // 00 11 01->1 10->0
        for (auto [y, id] : g[x])
        {
            if (y == pre)
                continue;
            dfs(y, x);
            if (id == 0)
            {
                cn1[y]++;
                ans += cn1[y] * 2 + cn2[y] + cn3[y];
                if (cn2[y] != 0 || cn4[y] != 0)
                {
                    cn3[y] += cn2[y];
                    if (cn2[y] == 0)
                        cn3[y]++;
                    cn2[y] = cn4[y] = 0;
                }
                ans += (cn1[x] * cn1[y] * 2 + cn1[x] * (cn3[y] + cn2[y]) + cn2[x] * cn2[y] * 2 + cn2[x] * (cn4[y] + cn1[y]) + cn3[x] * (cn1[y]) + cn4[x] * cn2[y]);
                cn1[x] += cn1[y], cn3[x] += cn3[y];
            }
            else
            {
                cn2[y]++, cn3[y] = 0;
                ans += cn2[y] * 2 + cn1[y] + cn4[y];
                if (cn1[y] != 0 || cn3[y] != 0)
                {
                    cn4[y] += cn1[y];
                    if (cn1[y] == 0)
                        cn4[y]++;
                    cn1[y] = cn3[y] = 0;
                }
                ans += (cn1[x] * cn1[y] * 2 + cn1[x] * (cn3[y] + cn2[y]) + cn2[x] * cn2[y] * 2 + cn2[x] * (cn4[y] + cn1[y]) + cn3[x] * (cn1[y]) + cn4[x] * cn2[y]);
                cn2[x] += cn2[y], cn4[x] += cn4[y];
            }
            // 00 11 01 10
        }
    };

    dfs(1, 0);
    // cout<<x1<<endl;
    cout << ans << endl;
}

// https://www.luogu.com.cn/problem/P2015
// 二叉苹果树 树上背包

void solve()
{
    int n, k;
    cin >> n >> k;

    vector<vector<array<int, 2>>> g(n + 1, vector<array<int, 2>>());
    for (int i = 1; i < n; i++)
    {
        int x, y, v;
        cin >> x >> y >> v;

        g[x].push_back({y, v});
        g[y].push_back({x, v});
    }

    vector<vector<int>> dp(n + 1, vector<int>(k + 1));
    function<void(int, int)> dfs = [&](int x, int pre)
    {
        for (auto [i, v] : g[x])
        {
            if (i == pre)
                continue;
            dfs(i, x);
            for (int j = k; j > 0; j--)
            {
                for (int l = j - 1; l >= 0; l--)
                {
                    dp[x][j] = max(dp[x][j], dp[i][l] + v + dp[x][j - l - 1]);
                }
            }
        }
    };

    dfs(1, 0);
    cout << dp[1][k] << endl;
}

// https://codeforces.com/problemset/problem/1637/F
//树D 最大值次大值计算贡献

void solve()
{
    int n, r = 1;
    cin >> n;

    vector<ll> h(n + 1);
    for (int i = 1; i <= n; i++)
    {
        cin >> h[i];
        if (h[i] >= h[r])
            r = i;
    }
    vector<vector<int>> g(n + 1, vector<int>());
    for (int i = 1; i < n; i++)
    {
        int x, y;
        cin >> x >> y;

        g[x].push_back(y);
        g[y].push_back(x);
    }

    ll ans = 0;
    vector<array<ll, 2>> dp(n + 1); // 子树节点中最大和次大
    function<void(int, int)> dfs = [&](int x, int pre)
    {
        for (auto i : g[x])
        {
            if (i == pre)
                continue;
            dfs(i, x);
            vector<ll> v;
            v.push_back(dp[x][0]), v.push_back(dp[x][1]);
            v.push_back(dp[i][0]);

            sort(all(v));
            dp[x][0] = v[2], dp[x][1] = v[1];
        }

        if (h[x] >= dp[x][0])
        {
            ans += h[x] - dp[x][0];
            dp[x][0] = h[x];
        }
        else if (h[x] > dp[x][1] && dp[x][1] != 0)
        {
            dp[x][1] = h[x];
        }
    };

    dfs(r, 0);
    ans += h[r] - dp[r][1];

    cout << ans << endl;
}

// https://atcoder.jp/contests/abc349/tasks/abc349_g
// 质因数分解+状压

void solve()
{
    ll n, m;
    cin >> n >> m;

    vector<array<ll, 2>> v;
    vector<ll> jc(N + 5);
    jc[0] = 1;
    for (int i = 1; i <= N; i++)
    {
        jc[i] = jc[i - 1] * 2;
        jc[i] %= mod;
    }

    ll tmp = m;
    for (ll i = 2; i <= tmp / i; i++)
    {
        if (tmp % i == 0)
        {
            ll cnt = 0;
            while (tmp % i == 0)
            {
                tmp /= i;
                cnt++;
            }
            v.push_back({i, cnt});
        }
    }
    if (tmp != 1)
        v.push_back({tmp, 1});

    int k = v.size();
    vector<ll> a(1 << k);
    vector<vector<ll>> dp(1 << k, vector<ll>(2));

    for (int i = 1; i <= n; i++)
    {
        ll x;
        cin >> x;

        if (m % x == 0)
        {
            ll now = 0;
            for (int j = 0; j < k; j++)
            {
                ll cnt = 0, y = v[j][0];
                while (x % y == 0)
                {
                    x /= y, cnt++;
                }
                if (cnt == v[j][1])
                {
                    now += (1 << j);
                }
            }
            a[now]++;
        }
    }

    if (m == 1)
    {
        cout << jc[a[0]] - 1 << endl;
        return;
    }
    dp[0][0] = 1;
    for (int i = 1; i <= a[0]; i++)
    {
        dp[0][0] *= 2;
        dp[0][0] %= mod;
    }

    ll now = 0;
    for (ll i = 1; i < (1 << k); i++)
    {
        if (a[i])
        {
            for (ll j = 0; j < (1 << k); j++)
            {
                dp[j][now ^ 1] = dp[j][now];
            }
            for (ll j = 0; j < (1 << k); j++)
            {
                dp[j | i][now ^ 1] += (dp[j][now] * (jc[a[i]] - 1) % mod);
                dp[j | i][now ^ 1] %= mod;
            }
            now ^= 1;
        }
    }

    cout << dp[(1 << k) - 1][now] % mod << endl;
}

// https://codeforces.com/contest/1914/problem/F
//树D 计数

void solve()
{
    int n;
    cin >> n;

    vector<vector<int>> g(n + 1, vector<int>());
    vector<int> siz(n + 1), son(n + 1);

    for (int i = 2; i <= n; i++)
    {
        int x;
        cin >> x;
        g[x].push_back(i);
    }

    int ans = 0;
    function<void(int)> dfs = [&](int x)
    {
        for (auto y : g[x])
        {
            dfs(y);
            siz[x] += siz[y];
            if (siz[son[x]] < siz[y])
                son[x] = y;
        }
        siz[x]++;
    };

    function<void(int)> dfs2 = [&](int x)
    {
        siz[x]--;
        if (siz[son[x]] <= siz[x] / 2)
        {
            ans += siz[x] / 2;
            siz[x] &= 1;
        }
        else
        {
            siz[x] -= siz[son[x]];
            dfs2(son[x]);
            siz[x] += siz[son[x]];
            int v = siz[son[x]] <= siz[x] - siz[son[x]] ? siz[x] / 2 : siz[x] - siz[son[x]];
            ans += v;
            siz[x] -= v * 2;
        }
        siz[x]++;
    };

    dfs(1);
    dfs2(1);

    cout << ans << endl;
}

// https://atcoder.jp/contests/abc350/tasks/abc350_g
//树上启发式合并 带权并查集

void solve()
{
    ll n, q, l = 0;
    cin >> n >> q;

    vector<int> f(n + 1), fa(n + 1), siz(n + 1);
    vector<vector<int>> g(n + 1, vector<int>());

    for (int i = 1; i <= n; i++)
    {
        fa[i] = i, siz[i] = 1;
    }

    function<void(int, int)> dfs = [&](int x, int pre)
    {
        f[x] = pre;
        vector<int> v;
        for (auto i : g[x])
        {
            if (i == pre)
                continue;
            v.push_back(i);
        }
        g[x].clear();

        g[x].push_back(pre), g[pre].push_back(x);
        for (auto i : v)
        {
            dfs(i, x);
        }
    };

    function<int(int)> find = [&](int x)
    {
        if (fa[x] == x)
            return x;
        return fa[x] = find(fa[x]);
    };

    while (q--)
    {
        ll op, x, y;
        cin >> op >> x >> y;

        op = 1 + (((op * (1 + l)) % mod) % 2);
        x = 1 + ((((x * (1 + l))) % mod) % n);
        y = 1 + ((((y * (1 + l))) % mod) % n);

        if (op == 1)
        {
            if (siz[find(x)] <= siz[find(y)])
            {
                siz[find(y)] += find(x);
                fa[find(x)] = find(y);
                dfs(x, y);
            }
            else
            {
                siz[find(x)] += find(y);
                fa[find(y)] = find(x);
                dfs(y, x);
            }
        }
        else
        {
            int ans = 0;
            if (f[f[x]] == y && f[x] != x && f[y] != y)
            {
                ans = f[x];
            }

            if (f[f[y]] == x && f[x] != x && f[y] != y)
            {
                ans = f[y];
            }

            if (f[y] == f[x] && f[x] != x && f[y] != y)
            {
                ans = f[x];
            }
            l = ans;

            cout << ans << endl;
        }
    }
}

// https://codeforces.com/gym/104354 E
//数字三角形 多维度滚动优化

void solve()
{
    int n, m, k;
    cin >> n >> m >> k;

    k = min(k, n * m);
    vector<vector<int>> a(n + 1, vector<int>(m + 1));
    for (int i = 1; i <= n; i++)
    {
        for (int j = 1; j <= m; j++)
        {
            char ch;
            cin >> ch;

            if (ch == '1')
                a[i][j] = 1;
            if (ch == '?')
                a[i][j] = -1;
        }
    }

    vector<vector<array<int, 2>>> dp(k + 1, vector<array<int, 2>>(m + 1, {-1, -1})); //?的数量 纵坐标 滚动;dp值表示当前状态的最大1的数量

    if (a[1][1] == 1)
        dp[0][1][1] = 1;
    if (a[1][1] == -1)
        dp[min(1, k)][1][1] = 0;
    if (a[1][1] == 0)
        dp[0][1][1] = 0;

    for (int i = 1; i <= n; i++)
    {
        int x1 = i & 1, x2 = (i & 1) ^ 1; // 本层，下一层
        for (int j = 1; j <= m; j++)
        {
            int id = 0, v = 0; // ？ 1
            if (a[i][j] == -1)
                id = 1;
            if (a[i][j] == 1)
                v = 1;
            for (int l = min(id, k); l <= k; l++)
            {
                int y = max(l - id, 0);
                if (dp[y][j - 1][x1] != -1)
                {
                    dp[l][j][x1] = max(dp[y][j - 1][x1] + v, dp[l][j][x1]);
                }
                if (dp[y][j][x2] != -1)
                {
                    dp[l][j][x1] = max(dp[y][j][x2] + v, dp[l][j][x1]);
                }

                if (id == 1 && l == k)
                {
                    if (dp[l][j - 1][x1] != -1)
                    {
                        dp[l][j][x1] = max(dp[l][j - 1][x1] + v, dp[l][j][x1]);
                    }
                    if (dp[l][j][x2] != -1)
                    {
                        dp[l][j][x1] = max(dp[l][j][x2] + v, dp[l][j][x1]);
                    }
                }
            }
        }

        for (int j = 0; j <= m; j++)
        {
            for (int l = 0; l <= k; l++)
            {
                dp[l][j][x2] = -1; // 清空上一层
            }
        }
    }

    int ans = 0;
    int x = n & 1;

    for (int l = 0; l <= k; l++)
    {
        for (int j = 1; j <= m; j++)
        {
            if (dp[l][j][x] != -1)
            {
                ans = max(ans, l + dp[l][j][x]);
            }
        }
    }

    cout << ans << endl;
}

// https://codeforces.com/problemset/problem/1771/D
//树上区间DP

void solve()
{
    int n;
    cin >> n;

    string s;
    cin >> s;
    s = " " + s;

    vector<vector<int>> g(n + 1, vector<int>());
    for (int i = 1; i < n; i++)
    {
        int x, y;
        cin >> x >> y;
        g[x].push_back(y);
        g[y].push_back(x);
    }

    vector<vector<int>> r(n + 1, vector<int>(n + 1)), dp(n + 1, vector<int>(n + 1));

    function<void(int, int, int)> dfs = [&](int rt, int x, int pre)
    {
        for (auto y : g[x])
        {
            if (y == pre)
                continue;
            dfs(rt, y, x);
            r[rt][y] = x;
        }
    };

    for (int i = 1; i <= n; i++)
    {
        dfs(i, i, 0);
    }

    function<int(int, int, int, int)> cal = [&](int rt1, int rt2, int x, int y)
    {
        if (dp[x][y])
            return dp[x][y];

        if (x == y)
        {
            return dp[x][y] = 1;
        }
        else if (r[rt2][x] == y)
        {
            return dp[x][y] = max(1, 2 * (s[x] == s[y]));
        }
        else
        {
            return dp[x][y] = max({cal(rt1, rt2, x, r[rt1][y]), cal(rt1, rt2, r[rt2][x], y), cal(rt1, rt2, r[rt2][x], r[rt1][y]) + 2 * (s[x] == s[y])});
        }
    };

    int ans = 1;
    for (int i = 1; i <= n; i++)
    {
        for (int j = 1; j <= n; j++)
        {
            ans = max(ans, cal(i, j, i, j));
            // cout<<i<<" "<<j<<" "<<cal(i,j)<<endl;
        }
    }

    cout << ans << endl;
}

// https://codeforces.com/gym/104090/problem/C
//前后缀背包

void solve()
{
    int n, k;
    cin >> n >> k;

    vector<vector<ll>> a(n + 2, vector<ll>(11));
    vector<ll> cnt(n + 2);

    ll sum = 0, val = 0;

    for (int i = 1; i <= n; i++)
    {
        cin >> cnt[i];
        for (int j = 1; j <= cnt[i]; j++)
        {
            cin >> a[i][j];
        }
        sum += cnt[i];
        val += a[i][cnt[i]];
    }

    if (sum <= k)
    {
        cout << val << endl;
        return;
    }

    vector<vector<ll>> dp1(n + 2, vector<ll>(k + 1)), dp2(n + 2, vector<ll>(k + 1));

    for (int i = 1; i <= n; i++)
    {
        for (int j = k; j >= 0; j--)
        {
            dp1[i][j] = dp1[i - 1][j];
        }

        for (int j = k; j - cnt[i] >= 0; j--)
        {
            if (dp1[i - 1][j - cnt[i]] != 0)
            {
                dp1[i][j] = max(dp1[i][j], dp1[i - 1][j - cnt[i]] + a[i][cnt[i]]);
            }
        }

        if (cnt[i] <= k)
            dp1[i][cnt[i]] = max(a[i][cnt[i]], dp1[i][cnt[i]]);
        else
            dp1[i][k] = max(dp1[i][k], a[i][k]);
    }

    for (int i = n; i >= 1; i--)
    {
        for (int j = k; j >= 0; j--)
        {
            dp2[i][j] = dp2[i + 1][j];
        }

        for (int j = k; j - cnt[i] >= 0; j--)
        {
            if (dp2[i + 1][j - cnt[i]] != 0)
            {
                dp2[i][j] = max(dp2[i][j], dp2[i + 1][j - cnt[i]] + a[i][cnt[i]]);
            }
        }
        if (cnt[i] <= k)
            dp2[i][cnt[i]] = max(a[i][cnt[i]], dp2[i][cnt[i]]);
        else
            dp2[i][k] = max(dp2[i][k], a[i][k]);
    }

    ll ans = 0;
    for (int i = 1; i <= n; i++)
    {
        for (int j = 0; j <= cnt[i] && j <= k; j++)
        {
            for (int l = 0; l <= k - j; l++)
            {
                if (dp1[i - 1][l] != 0 && dp2[i + 1][k - j - l] != 0)
                    ans = max(ans, dp1[i - 1][l] + dp2[i + 1][k - j - l] + a[i][j]);
            }

            if (dp1[i - 1][k - j] != 0)
            {
                ans = max(ans, dp1[i - 1][k - j] + a[i][j]);
            }
            if (dp2[i + 1][k - j] != 0)
            {
                ans = max(ans, dp2[i + 1][k - j] + a[i][j]);
            }
            if (k - j == 0)
            {
                ans = max(ans, a[i][j]);
            }
        }
    }

    ans = max(ans, dp1[n][k]);
    ans = max(ans, dp2[1][k]);

    cout << ans << endl;
}

// https://codeforces.com/contest/1858/problem/D
// 双指针dp

void solve()
{
    int n, k;
    cin >> n >> k;

    string s;
    cin >> s;

    s = " " + s;

    vector<vector<int>> l1(n + 2, vector<int>(k + 1)), l0(n + 2, vector<int>(k + 1)), r1(n + 2, vector<int>(k + 1)), r0(n + 2, vector<int>(k + 1));

    for (int i = 1; i <= n; i++)
    {
        int p1 = i, p0 = i;
        for (int j = 0; j <= k; j++)
        {
            while (p1 >= 1 && s[p1] == '1')
                p1--;
            l1[i][j] = max(i - p1, l1[i - 1][j]);
            if (p1 >= 1)
                p1--;
        }
        for (int j = 0; j <= k; j++)
        {
            while (p0 >= 1 && s[p0] == '0')
                p0--;
            l0[i][j] = max(i - p0, l0[i - 1][j]);
            if (p0 >= 1)
                p0--;
        }
    }

    for (int i = n; i >= 1; i--)
    {
        int p1 = i, p0 = i;
        for (int j = 0; j <= k; j++)
        {
            while (p1 <= n && s[p1] == '1')
                p1++;
            r1[i][j] = max(p1 - i, r1[i + 1][j]);
            if (p1 <= n)
                p1++;
        }
        for (int j = 0; j <= k; j++)
        {
            while (p0 <= n && s[p0] == '0')
                p0++;
            r0[i][j] = max(p0 - i, r0[i + 1][j]);
            if (p0 <= n)
                p0++;
        }
    }

    vector<int> dp(n + 1, -1); // 存每个0对应的最大1
    for (int i = 0; i <= n; i++)
    {
        for (int j = 0; j <= k; j++)
        {
            dp[l0[i][j]] = max(dp[l0[i][j]], r1[i + 1][k - j]);
            dp[r0[i + 1][j]] = max(dp[r0[i + 1][j]], l1[i][k - j]);
        }
    }

    for (int i = 1; i <= n; i++)
    {
        int res = 0;
        for (int j = 0; j <= n; j++)
        {
            if (dp[j] != -1)
                res = max(res, i * j + dp[j]);
        }

        cout << res << " \n"[i == n];
    }
}

// https://ac.nowcoder.com/acm/contest/81597/G
//状压+背包 质数性质

void solve()
{
    vector<int> p; // 11个
    for (int i = 2; i <= 33; i++)
    {
        int x = i;
        for (int j = 2; j * j <= x; j++)
        {
            if (x % j == 0)
                x /= j;
        }

        if (x == i)
            p.push_back(i);
    }

    int n;
    cin >> n;

    vector<array<ll, 2>> a(n + 1);

    for (int i = 1; i <= n; i++)
    {
        cin >> a[i][0];
        a[i][1] = 1;
        for (int j = 2; j * j <= a[i][0]; j++)
        {
            while (a[i][0] % (j * j) == 0)
            {
                a[i][0] /= (j * j);
                a[i][1] *= j;
            }
        }
    }

    vector<ll> v1;          // 记录1的贡献
    vector<array<ll, 3>> w; // 大质数 状态 权值
    vector<ll> val(1 << 12, 1);

    for (int i = 1; i < (1 << 12); i++)
    {
        for (int j = 0; j <= 11; j++)
        {
            if (i & (1 << j))
                val[i] *= p[j];
        }
    }

    for (int i = 1; i <= n; i++)
    {
        if (a[i][0] == 1)
        {
            v1.push_back(a[i][1]);
            continue;
        }

        int x = a[i][0], y = 0;
        for (int j = 0; j < p.size(); j++)
        {
            if (x % p[j] == 0)
            {
                y += (1 << j);
                x /= p[j];
            }
        }
        w.push_back({x, y, a[i][1]});
    }

    sort(all(w));

    vector<vector<ll>> dp(1001, vector<ll>(1 << 12));
    // dp[1][0] = 1;

    for (auto [x, y, v] : w)
    {
        if (x == 1)
        {
            vector<ll> vc(1 << 12);
            for (int i = 0; i < (1 << 12); i++)
            {
                vc[i ^ y] += dp[1][i] * val[i & y] % mod * v % mod;
                vc[i ^ y] %= mod;
            }

            for (int i = 0; i < (1 << 12); i++)
            {
                dp[1][i] += vc[i] % mod;
                dp[1][i] %= mod;
            }
            dp[1][y] += v;
            dp[1][y] %= mod;
            // cout<<dp[1][0]<<endl;
        }
        else
        {
            vector<ll> vc1(1 << 12), vc2(1 << 12);
            for (int i = 0; i < (1 << 12); i++)
            {
                vc1[i ^ y] += dp[x][i] * val[i & y] % mod * v % mod * x % mod;
                vc1[i ^ y] %= mod;
            }

            for (int i = 0; i < (1 << 12); i++)
            {
                vc2[i ^ y] += dp[1][i] * val[i & y] % mod * v % mod;
                vc2[i ^ y] %= mod;
            }

            for (int i = 0; i < (1 << 12); i++)
            {
                dp[1][i] += vc1[i];
                dp[1][i] %= mod;
                dp[x][i] += vc2[i];
                dp[x][i] %= mod;
            }
            dp[x][y] += v;
            dp[x][y] %= mod;
        }
    }

    ll ans = dp[1][0];
    for (auto i : v1)
    {
        ans = ans + ans * i % mod + i;
        ans %= mod;
    }

    cout << ans << endl;
}

// https://www.luogu.com.cn/problem/P1912
//决策单调性，二分队列

void solve()
{
    ll n, L, p;
    cin >> n >> L >> p;

    vector<string> s(n + 1);
    for (int i = 1; i <= n; i++)
        cin >> s[i];
    vector<ll> pre(n + 1);
    for (int i = 1; i <= n; i++)
    {
        pre[i] = s[i].size() + pre[i - 1];
    }

    vector<long double> dp(n + 1);
    vector<int> opt(n + 1);

    deque<array<int, 3>> dq; // id l r
    dq.push_back({0, 1, n});

    function<long double(int, int)> cal = [&](int x, int y)
    {
        long double res = 1;
        for (int i = 1; i <= p; i++)
            res *= abs(pre[y] - pre[x] + y - x - 1 - L);
        return dp[x] + res;
    };

    for (int i = 1; i <= n; i++)
    {
        while (!dq.empty() && dq.front()[2] < i)
            dq.pop_front();
        dp[i] = cal(dq.front()[0], i), opt[i] = dq.front()[0];

        while (!dq.empty() && cal(dq.back()[0], dq.back()[1]) >= cal(i, dq.back()[1]))
            dq.pop_back();

        if (dq.empty())
        {
            dq.push_back({i, i + 1, n});
        }
        else if (cal(dq.back()[0], dq.back()[2]) < cal(i, dq.back()[2]))
        {
            if (dq.back()[2] != n)
            {
                dq.push_back({i, dq.back()[2] + 1, n});
            }
        }
        else
        {
            int l = dq.back()[1], r = n;
            while (l < r)
            {
                int mid = (l + r) / 2;
                if (cal(dq.back()[0], mid) >= cal(i, mid))
                {
                    r = mid;
                }
                else
                {
                    l = mid + 1;
                }
            }

            dq.back()[2] = l - 1;
            dq.push_back({i, l, n});
        }
    }

    if (dp[n] > 1e18)
    {
        cout << "Too hard to arrange" << endl;
    }
    else
    {
        cout << (ll)dp[n] << endl;
        stack<array<int, 2>> st;
        int x = n;
        while (x)
        {
            st.push({opt[x] + 1, x});
            x = opt[x];
        }

        while (!st.empty())
        {
            auto [l, r] = st.top();
            st.pop();
            for (int i = l; i <= r; i++)
            {
                cout << s[i] << " \n"[i == r];
            }
        }
    }

    cout << "--------------------" << endl;
}

// https://codeforces.com/contest/321/problem/E
// 决策单调性
void solve()
{
    int n, k;
    cin >> n >> k;

    vector<vector<ll>> a(n + 1, vector<ll>(n + 1)), pre(n + 1, vector<ll>(n + 1));
    string s;
    getline(cin, s);
    for (int i = 1; i <= n; i++)
    {

        getline(cin, s);
        int p = 0;
        for (int j = 1; j <= n; j++)
        {
            // cin>>a[i][j];
            a[i][j] = s[p] - '0';
            p += 2;
            pre[i][j] = pre[i - 1][j] + pre[i][j - 1] + a[i][j] - pre[i - 1][j - 1];
        }
    }

    vector<vector<ll>> dp(n + 1, vector<ll>(k + 1, INF));
    dp[0][0] = 0;

    function<ll(int, int, int)> cal = [&](int x, int y, int j)
    {
        ll res = 0;
        res += pre[y][y] - pre[x][y] - pre[y][x] + pre[x][x];
        return dp[x][j - 1] + res / 2;
    };

    for (int j = 1; j <= k; j++)
    {
        deque<array<int, 3>> dq; // id l r
        dq.push_back({0, 1, n});
        for (int i = 1; i <= n; i++)
        {
            while (!dq.empty() && dq.front()[2] < i)
                dq.pop_front();
            dp[i][j] = cal(dq.front()[0], i, j);

            while (!dq.empty() && cal(dq.back()[0], dq.back()[1], j) >= cal(i, dq.back()[1], j))
                dq.pop_back();

            if (dq.empty())
            {
                dq.push_back({i, i + 1, n});
            }
            else if (cal(dq.back()[0], dq.back()[2], j) < cal(i, dq.back()[2], j))
            {
                if (dq.back()[2] != n)
                {
                    dq.push_back({i, dq.back()[2] + 1, n});
                }
            }
            else
            {
                int l = dq.back()[1], r = n;
                while (l < r)
                {
                    int mid = (l + r) / 2;
                    if (cal(dq.back()[0], mid, j) >= cal(i, mid, j))
                    {
                        r = mid;
                    }
                    else
                    {
                        l = mid + 1;
                    }
                }

                dq.back()[2] = l - 1;
                dq.push_back({i, l, n});
            }
        }
    }

    cout << dp[n][k] << endl;
}

// https://codeforces.com/gym/105336/attachments
// CCPC网络赛I题，将贡献数组转换成前缀和的形式，最后用差分还原

void solve()
{
    ll n, m;
    cin >> n >> m;

    vector<ll> a(n + 1), b(m + 1);
    for (int i = 1; i <= n; i++)
    {
        cin >> a[i];
    }
    for (int i = 1; i <= m; i++)
    {
        cin >> b[i];
    }
    sort(all(a)), sort(all(b));
    if (a[1] > b[m])
    {
        cout << 0 << endl;
        return;
    }

    ll ans = 0, n1 = b[m] - a[1];
    vector<ll> sum(n1 + 2);
    for (int t = 0; t <= n1; t++)
    {
        vector<vector<ll>> dp(m + 1, vector<ll>(n + 1));
        ll p = 0;
        for (int i = 0; i <= m; i++)
            dp[i][0] = 1;

        for (int i = 1; i <= m; i++)
        {
            while (p < n && (a[p + 1] <= b[i]))
                p++;
            for (int j = 1; j <= p; j++)
            {
                dp[i][j] = dp[i - 1][j] + dp[i - 1][j - 1] * (p - j + 1) % mod;
                dp[i][j] %= mod;
            }
        }

        for (int j = 1; j <= n; j++)
        {
            sum[t] = (sum[t] + dp[m][j]) % mod;
        }

        for (int i = 1; i <= n; i++)
        {
            a[i]++;
        }
    }

    for (int i = 0; i <= n1; i++)
    {
        ans = (ans + (sum[i] - sum[i + 1]) * i % mod + mod) % mod;
    }

    cout << ans << endl;
}

// https://codeforces.com/contest/2005/problem/E1
// 博弈->记搜

void solve()
{
    int l, n, m;
    cin >> l >> n >> m;

    vector<int> a(l + 1);
    vector<vector<int>> g(n + 1, vector<int>(m + 1));
    vector<vector<array<int, 2>>> st(8);

    for (int i = 1; i <= l; i++)
        cin >> a[i];
    for (int i = 1; i <= n; i++)
    {
        for (int j = 1; j <= m; j++)
        {
            cin >> g[i][j];
            while (!st[g[i][j]].empty() && st[g[i][j]].back()[1] < j)
                st[g[i][j]].pop_back();
            st[g[i][j]].push_back({i, j});
        }
    }

    vector<vector<vector<int>>> dp(n + 1, vector<vector<int>>(m + 1, vector<int>(l + 1, -1)));
    function<bool(int, int, int)> dfs = [&](int x, int y, int k)
    {
        if (dp[x][y][k] != -1)
            return dp[x][y][k];
        if (k == l)
            return dp[x][y][k] = 1;

        dp[x][y][k] = 1;
        for (auto [x1, y1] : st[a[k + 1]])
        {
            if (x1 > x && y1 > y)
            {
                if (dfs(x1, y1, k + 1))
                    dp[x][y][k] = 0;
            }
        }

        return dp[x][y][k];
    };

    for (int i = 1; i <= n; i++)
    {
        for (int j = 1; j <= m; j++)
        {
            if (g[i][j] == a[1])
            {
                if (dfs(i, j, 1))
                {
                    cout << "T" << endl;
                    return;
                }
            }
        }
    }

    cout << "N" << endl;
}

// https://atcoder.jp/contests/abc373/tasks/abc373_f
// 背包+调和级数

ll n, w;
cin >> n >> w;

vector<array<ll, 2>> a(n + 1);
vector<priority_queue<array<ll, 3>>> q(w + 1);
for (int i = 1; i <= n; i++)
{
    cin >> a[i][0] >> a[i][1];
    q[a[i][0]].push({a[i][1] - 1, i, 1});
}

vector<ll> dp(w + 1);

ll ans = 0;
vector<ll> jc(w + 1), pre(w + 1);
jc[1] = 1, pre[1] = 1;
for (ll i = 2; i <= w; i++)
{
    jc[i] = i * i - pre[i - 1];
    pre[i] = pre[i - 1] + jc[i];
}

for (int i = 1; i <= w; i++)
{
    for (int j = 1; j <= w / i + 1 && !q[i].empty(); j++)
    {
        auto [v, id, now] = q[i].top();
        q[i].pop();

        for (int i1 = w; i1 - a[id][0] * now >= 0; i1--)
        {
            dp[i1] = max(dp[i1], dp[i1 - a[id][0]] + v);
            ans = max(ans, dp[i1]);
        }
        if (now + 1 <= w)
            q[i].push({a[id][1] - jc[now + 1], id, now + 1});
    }
}

cout << ans << endl;

// https://atcoder.jp/contests/abc369/tasks/abc369_f
// DS优化DP

void solve()
{
    int n, m, k;
    cin >> n >> m >> k;

    vector<array<int, 3>> a(k + 1);
    vector<array<int, 2>> mp(k + 1);
    mp[0] = {1, 1};
    for (int i = 1; i <= k; i++)
    {
        cin >> a[i][0] >> a[i][1];
        a[i][2] = i;
        mp[i] = {a[i][0], a[i][1]};
    }

    sort(all(a));
    vector<int> dp(k + 1), pre(k + 1);

    Bit bt(m + 10);

    vector<int> id(k + 1);

    int ans = 0;
    for (int i = 1; i <= k; i++)
    {
        dp[i] = bt.query(a[i][1]) + 1;
        pre[i] = id[bt.query(a[i][1])];
        bt.add(a[i][1], dp[i]);
        if (dp[i] > dp[ans])
            ans = i;
        if (id[dp[i]] == 0)
        {
            id[dp[i]] = i;
        }
        else
        {
            if (a[id[dp[i]]][1] > a[i][1])
            {
                id[dp[i]] = i;
            }
        }
    }

    cout << dp[ans] << endl;

    stack<char> st;
    for (int i = mp[a[ans][2]][0]; i < n; i++)
    {
        st.push('D');
    }
    for (int i = mp[a[ans][2]][1]; i < m; i++)
    {
        st.push('R');
    }
    while (ans != 0)
    {
        int p = pre[ans];
        for (int i = mp[a[p][2]][0]; i < mp[a[ans][2]][0]; i++)
        {
            st.push('D');
        }
        for (int i = mp[a[p][2]][1]; i < mp[a[ans][2]][1]; i++)
        {
            st.push('R');
        }
        ans = p;
    }

    while (!st.empty())
    {
        cout << st.top();
        st.pop();
    }
    cout << endl;
}

// https://codeforces.com/gym/104128/problem/B
// 单调队列优化桥墩问题，查询时只关心修改点的后k个，预处理前后缀的贡献

void solve()
{
    ll n, k;
    cin >> n >> k;

    vector<ll> a(n + 2);
    for (int i = 1; i <= n; i++)
    {
        cin >> a[i];
    }

    string s;
    cin >> s;
    s = " " + s;

    vector<ll> dp(n + 2), suf(n + 2);
    deque<int> dq;
    dq.push_back(0);

    for (int i = 1; i <= n; i++)
    {
        while (!dq.empty() && i - dq.front() > k)
            dq.pop_front();
        dp[i] = dp[dq.front()] + a[i];
        if (s[i] == '1')
            dq.clear(); // must
        while (!dq.empty() && dp[dq.back()] > dp[i])
            dq.pop_back();
        dq.push_back(i);
    }

    dq.clear();
    dq.push_back(n + 1);

    for (int i = n; i >= 1; i--)
    {
        while (!dq.empty() && dq.front() - i > k)
            dq.pop_front();
        suf[i] = suf[dq.front()] + a[i];
        if (s[i] == '1')
            dq.clear(); // must
        while (!dq.empty() && suf[dq.back()] > suf[i])
            dq.pop_back();
        dq.push_back(i);
    }

    int q;
    cin >> q;

    while (q--)
    {
        ll p, v;
        cin >> p >> v;
        deque<array<ll, 2>> dq1; // val id

        int l = max(0ll, p - k), r = min(n + 1, p + k);
        swap(a[p], v);
        dq1.push_back({dp[l], l});
        for (int i = l + 1; i <= p; i++)
        {
            while (!dq1.empty() && i - dq1.front()[1] > k)
                dq1.pop_front();
            ll x = dq1.front()[0] + a[i];
            if (i < p)
                x = dp[i];
            if (s[i] == '1')
                dq1.clear(); // must
            while (!dq1.empty() && dq1.back()[0] > x)
                dq1.pop_back();
            dq1.push_back({x, i});
        }
        swap(a[p], v);
        ll res = dq1.back()[0] + suf[p] - a[p];
        for (int i = p + 1; i <= r; i++)
        {
            while (!dq1.empty() && i - dq1.front()[1] > k)
                dq1.pop_front();
            ll x = dq1.front()[0] + a[i];
            if (s[i] == '1')
                dq1.clear(); // must
            while (!dq1.empty() && dq1.back()[0] > x)
                dq1.pop_back();
            dq1.push_back({x, i});
            res = min(res, x + suf[i] - a[i]);
        }

        cout << res << endl;
    }
}

// https://atcoder.jp/contests/abc374/tasks/abc374_f
// 拆点DP

void solve()
{
    ll n, k, x;
    cin >> n >> k >> x;
    vector<ll> a(n + 1), b(n * n + 1);
    ll sum = 0, ans = INF;
    for (int i = 1; i <= n; i++)
    {
        cin >> a[i];
        sum += a[i];
        for (int j = 1; j <= n; j++)
        {
            b[(i - 1) * n + j] = a[i] + (j - 1) * x;
        }
    }

    sort(b.begin() + 1, b.end());
    int p1 = 1, p2 = 0; // p1指向当前时刻最多能够发多少货 p2指向当前能够转移的最靠后的点

    vector<vector<ll>> dp(n * n + 1, vector<ll>(n + 1, INF)); // dp i,j 表示在b[i]时刻已经发了j个货的最小代价
    dp[0][0] = 0;

    for (int i = 1; i <= n * n; i++)
    {
        for (int j = 0; j <= n; j++)
        {
            dp[i][j] = dp[i - 1][j];
        }

        while (p1 + 1 <= n && a[p1 + 1] <= b[i])
            p1++;
        while (b[p2 + 1] + x <= b[i])
            p2++;

        for (int j = 0; j <= p1; j++)
        {
            for (int j1 = 1; j1 <= min(k, (ll)j); j1++)
            {
                dp[i][j] = min(dp[i][j], dp[p2][j - j1] + j1 * b[i]);
            }
        }
        ans = min(ans, dp[i][n]);
    }

    cout << ans - sum << endl;
}

// https://codeforces.com/contest/2025/problem/E
// f i,j -> 考虑第i大的数分配给了A B且 A多出来j个 or A少j个的方案数

void solve()
{
    ll n, m;
    cin >> n >> m;

    vector<vector<ll>> f(m + 2, vector<ll>(m + 2));
    f[1][1] = 1;
    for (int i = 2; i <= m; i++)
    {
        for (int j = 0; j <= i; j++)
        {
            if (j > 0)
                f[i][j] += f[i - 1][j - 1];
            f[i][j] += f[i - 1][j + 1];
            f[i][j] %= mod;
        }
    }

    ll ans = 0;
    vector<vector<ll>> dp(n + 1, vector<ll>(m + 1));

    dp[0][0] = 1;
    for (int i = 1; i <= n - 1; i++)
    {
        for (int j1 = m; j1 >= 0; j1 -= 2)
        {
            for (int j2 = 0; j2 <= j1; j2 += 2)
            {
                dp[i][j1] += dp[i - 1][j1 - j2] * f[m][j2] % mod;
                dp[i][j1] %= mod;
            }
        }
    }

    for (int i = 0; i <= m; i += 2)
    {
        ans += dp[n - 1][i] * f[m][i] % mod;
        ans %= mod;
    }

    cout << ans << endl;
}

//南京2021 H
//树上DP 但是很屎


void solve()
{
    int n;
    cin >> n;

    vector<ll> a(n + 1), b(n + 1);
    for (int i = 1; i <= n; i++)
    {
        cin >> a[i];
    }
    for (int i = 1; i <= n; i++)
    {
        cin >> b[i];
    }
    vector<vector<int>> g(n + 1);

    for (int i = 1; i < n; i++)
    {
        int x, y;
        cin >> x >> y;
        g[x].push_back(y);
        g[y].push_back(x);
    }

    vector<array<ll, 2>> dp(n + 1); // 0表示马上拿 1表示后面拿

    function<void(int, int)> dfs = [&](int x, int f)
    {
        ll sum = a[x], ma = 0;
        ll ma2 = 0, ma3 = 0, ma2_id = 0, ma3_id = 0;
        ll cma2 = 0, cma3 = 0, cma2_id = 0, cma3_id = 0;
        for (auto y : g[x])
        {
            if (y == f)
                continue;
            dfs(y, x);
            sum += dp[y][0] - a[y];
            ma = max(ma, a[y]);
            if (b[y] == 3)
            {
                if (a[y] > ma2)
                {
                    cma2_id = ma2_id;
                    cma2 = ma2;
                    ma2_id = y;
                    ma2 = a[y];
                }
                else if (a[y] > cma2)
                {
                    cma2_id = y;
                    cma2 = a[y];
                }
            }
            if (dp[y][1] - (dp[y][0] - a[y]) > ma3)
            {
                cma3_id = ma3_id;
                cma3 = ma3;
                ma3_id = y;
                ma3 = dp[y][1] - (dp[y][0] - a[y]);
            }
            else if (dp[y][1] - (dp[y][0] - a[y]) > cma3)
            {
                cma3_id = y;
                cma3 = dp[y][1] - (dp[y][0] - a[y]);
            }
        }
        dp[x][1] = sum;
        dp[x][0] = sum + ma;
        if (ma2_id != ma3_id)
        {
            dp[x][0] = max(dp[x][0], sum + ma2 + ma3);
        }
        else
        {
            dp[x][0] = max(dp[x][0], sum + max(ma2 + cma3, cma2 + ma3));
        }
    };
    dfs(1, 0);
    cout << dp[1][0] << endl;
}